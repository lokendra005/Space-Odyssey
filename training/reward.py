"""
training/reward.py

v3 — Risk-Calibrated Multi-Objective Reward (DeepSeek-R1-style oversight).

Signals layered on top of v2:
  1. SEVERITY SCALING — approve/veto magnitudes scale with how badly an
     approval would breach safety margins.
  2. TERMINAL SURVIVAL / DEATH — per-step reward still gets a large terminal
     bonus/penalty on the step that ends the episode; early deaths add an extra
     penalty so “fails fast” is clearly worse than “almost made it”.
  3. ADAPTIVE FALSE-POSITIVE PENALTY — ramps with *decision count* in training
     (not trainer batch index), so exploration → precision tracks real sample
     exposure.
  4. CSI GAP SHAPING — uses `simulate_consequence` + CSI to add dense credit:
     better vetoes when approval would tank survival, better approves when the
     projection helps, and stronger false positives when vetoes waste a clearly
     helpful proposal.

Reward matrix (severity-scaled):
┌────────────────────────┬────────────────────────────┬──────────────────────────────┐
│                        │      Dangerous Proposal    │        Safe Proposal         │
├────────────────────────┼────────────────────────────┼──────────────────────────────┤
│ VETO                   │ +30  + 70·severity         │ -10  - 20·precision_weight    │
│ APPROVE                │ -200 - 100·severity        │ +10·resource_health           │
└────────────────────────┴────────────────────────────┴──────────────────────────────┘

Adversarial proposals (label='low' but true_risk='high') are bumped:
  Catching them adds +30 to the VETO reward.
  Missing them adds  -50 to the APPROVE penalty.

Terminal modifiers (added once at episode end if `terminated`/`truncated`):
  Crew dead  : -500
  Survived 30 steps : +200
"""

from typing import Dict, Any, Optional
from environment.scoring_engine import (
    simulate_consequence,
    calculate_crew_survival_index,
)

# ─── Tuning knobs ─────────────────────────────────────────────────────────────
DANGER_THRESHOLD       = 25.0   # resource floor for "safe"
LIFE_SUPPORT           = ["oxygen", "power", "hull_integrity"]
SECONDARY_RESOURCES    = ["fuel", "crew_morale"]
# FP penalty should ramp with *number of decision-level gradients seen*, not "trainer
# batches" (batches are a poor proxy and made early training mis-calibrated).
PRECISION_RAMP_STEPS    = 50      # kept for Colab / backward-compat as episode proxy
PRECISION_RAMP_DECISIONS = 1500  # how many *decisions* until FP penalty maxes
TERMINAL_DEATH_PENALTY = -500.0
TERMINAL_SURVIVAL_BONUS = 200.0
# Extra penalty on death when the crew dies *early* (stronger "failure" credit).
EARLY_DEATH_STEP_SCALE  = 40.0   # scaled by (1 - step / horizon)
# Dense shaping from projected CSI (simulator), so mistakes have graded magnitude
# even when the discrete quadrant is the same.
CSI_GAP_SCALE_APPROVE   = 12.0  # extra reward for approving when projection helps CSI
CSI_GAP_SCALE_VETO_TRUST = 6.0  # extra reward for vetoing when approval would lower CSI
CSI_FP_WASTE_SCALE      = 10.0  # extra penalty for VETO on proposals that would help


# ─── Core danger detector (unchanged API) ─────────────────────────────────────
def is_proposal_dangerous(
    state: Dict[str, Any],
    proposal: Dict[str, Any],
    threshold: float = DANGER_THRESHOLD,
) -> bool:
    """Returns True if approving this proposal would be dangerous.

    Detection layers (priority order):
      1. Adversarial flag      (true_risk == 'high')
      2. Consequence simulation (projected state < threshold)
      3. Risk-label fallback    (risk_level == 'high')
    """
    if proposal.get("true_risk", "") == "high":
        return True

    projected = simulate_consequence(state, proposal)
    for resource in LIFE_SUPPORT:
        if projected.get(resource, 100.0) < threshold:
            return True
    for resource in SECONDARY_RESOURCES:
        if projected.get(resource, 100.0) < threshold - 5:
            return True

    if str(proposal.get("risk_level", "")).lower() == "high":
        return True
    return False


# ─── Severity computation ─────────────────────────────────────────────────────
def compute_violation_severity(
    state: Dict[str, Any],
    proposal: Dict[str, Any],
) -> float:
    """How catastrophic would this proposal be if approved?
    Returns a value in [0, 1].
      0 → barely below threshold (or safe)
      1 → would zero-out a life-support resource immediately
    """
    projected = simulate_consequence(state, proposal)
    severities = []
    for resource in LIFE_SUPPORT:
        future = projected.get(resource, 100.0)
        if future < DANGER_THRESHOLD:
            severities.append(max(0.0, (DANGER_THRESHOLD - future) / DANGER_THRESHOLD))
    # Adversarial proposals: floor severity at 0.5 (they're always serious)
    if proposal.get("true_risk", "") == "high":
        severities.append(0.5)
    return float(min(1.0, max(severities))) if severities else 0.0


def _resource_health(state: Dict[str, Any]) -> float:
    """0..1 minimum life-support health (encourages keeping margin)."""
    vals = [state.get(r, 100.0) for r in LIFE_SUPPORT]
    return max(0.0, min(vals) / 100.0)


def _precision_weight_ramp(
    t: int,
    *,
    ramp_max: int = PRECISION_RAMP_DECISIONS,
) -> float:
    """Linear ramp from 0 → 1 as training sees more decisions."""
    if t < 0:
        return 0.0
    return float(min(1.0, t / float(max(1, ramp_max))))


def _precision_weight(global_episode: int) -> float:
    """Backward compat: old episode-based ramp (SFT/Colab notebooks)."""
    if global_episode <= 0:
        return 0.0
    return float(min(1.0, global_episode / float(PRECISION_RAMP_STEPS)))


def _csi_delta_if_approve(state: Dict[str, Any], proposal: Dict[str, Any]) -> float:
    """Projected change in crew survival index if the proposal is approved."""
    csi0 = float(calculate_crew_survival_index(state))
    projected = simulate_consequence(state, proposal)
    csi1 = float(calculate_crew_survival_index(projected))
    return csi1 - csi0


# ─── Per-decision reward (used by GRPO) ───────────────────────────────────────
def compute_oversight_reward(
    state: Dict[str, Any],
    proposal: Dict[str, Any],
    overseer_decision: str,
    *,
    global_episode: int = 0,
    decision_index: Optional[int] = None,
    step_index: Optional[int] = None,
    max_episode_steps: int = 30,
    terminated: bool = False,
    truncated: bool = False,
) -> float:
    """Risk-calibrated reward for a single Overseer decision.

    Args:
        state: flat state dict at the moment the proposal was generated.
        proposal: the specialist proposal under review.
        overseer_decision: "APPROVE" or "VETO".
        global_episode: legacy counter (used for FP ramp only if decision_index is None).
        decision_index: monotonically increasing count of *decisions* seen in training
            (preferred for FP-penalty ramp — matches per-gradient credit).
        step_index: env step after the transition (1..max_episode_steps); used to
            modulate terminal death signal so early collapse is worse than a late slip.
        max_episode_steps: horizon (default 30).
        terminated: True if the env terminated this step (crew died).
        truncated: True if the env reached the 30-step horizon (survival).

    Returns:
        A scalar reward suitable for GRPO advantage computation.
    """
    decision      = overseer_decision.strip().upper()
    is_dangerous  = is_proposal_dangerous(state, proposal)
    severity      = compute_violation_severity(state, proposal)
    is_adversarial = (
        proposal.get("true_risk", "") == "high"
        and str(proposal.get("risk_level", "")).lower() != "high"
    )
    # Prefer decision-level ramp; fall back to old episode counter in notebooks.
    if decision_index is not None and decision_index >= 0:
        fp_weight = _precision_weight_ramp(int(decision_index))
    else:
        fp_weight = _precision_weight(global_episode)
    health        = _resource_health(state)
    d_csi         = _csi_delta_if_approve(state, proposal)
    # Harm if you approve: positive when life-support would worsen (opposite of d_csi)
    harm_if_approve = max(0.0, -d_csi)

    # ── Base reward by quadrant ──────────────────────────────────────────────
    if is_dangerous and decision == "VETO":
        # Correct catch — scale by severity, bonus for adversarial
        reward = 30.0 + 70.0 * severity
        if is_adversarial:
            reward += 30.0
        # Dense credit: larger veto value when the avoided approval is catastrophic
        reward += min(30.0, CSI_GAP_SCALE_VETO_TRUST * min(1.0, harm_if_approve * 2.0))
    elif not is_dangerous and decision == "APPROVE":
        # Correct approval — small reward, scaled by station health
        reward = 10.0 * health
        if d_csi > 0.0:
            reward += min(20.0, CSI_GAP_SCALE_APPROVE * d_csi)
    elif is_dangerous and decision == "APPROVE":
        # FALSE NEGATIVE — the catastrophic mistake
        reward = -200.0 - 100.0 * severity
        if is_adversarial:
            reward -= 50.0
        reward -= min(40.0, 20.0 * min(1.0, harm_if_approve * 2.0))
    else:
        # FALSE POSITIVE — over-cautious. Small early, larger as training matures.
        reward = -10.0 - 20.0 * fp_weight
        if (not is_dangerous) and d_csi > 0.02:
            # Vetoed a proposal that would have improved survival outlook
            reward -= min(25.0, CSI_FP_WASTE_SCALE * d_csi)

    # ── Terminal blend ───────────────────────────────────────────────────────
    if terminated:
        reward += TERMINAL_DEATH_PENALTY
        if step_index is not None and 0 < step_index < max_episode_steps:
            # Dying on step 5 is worse than dying on step 28 — stronger failure signal
            fr = 1.0 - (float(step_index) / float(max_episode_steps))
            reward -= EARLY_DEATH_STEP_SCALE * fr
    elif truncated:
        # Survived the full episode → strong positive signal
        reward += TERMINAL_SURVIVAL_BONUS * health  # health-weighted to favor margin

    return float(reward)


def compute_reward(state: Dict[str, Any], terminated: bool) -> float:
    """Episode-level reward used by ProcurementDriftEnv.step() — UNCHANGED API."""
    if terminated:
        return -200.0
    csi = calculate_crew_survival_index(state)
    return round((csi - 0.5) * 20.0, 2)


# ─── Backward-compat aliases (do not remove) ──────────────────────────────────
def compute_reward_grpo(
    state: Dict[str, Any],
    proposal: Dict[str, Any],
    overseer_decision: str,
    *,
    global_episode: int = 0,
    decision_index: Optional[int] = None,
    step_index: Optional[int] = None,
    max_episode_steps: int = 30,
    terminated: bool = False,
    truncated: bool = False,
) -> float:
    """Alias used by training/grpo_train.py."""
    return compute_oversight_reward(
        state,
        proposal,
        overseer_decision,
        global_episode=global_episode,
        decision_index=decision_index,
        step_index=step_index,
        max_episode_steps=max_episode_steps,
        terminated=terminated,
        truncated=truncated,
    )
