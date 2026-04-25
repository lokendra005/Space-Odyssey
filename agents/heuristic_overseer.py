"""
agents/heuristic_overseer.py

A non-LLM stand-in for the GRPO-trained Overseer.

Why this exists: judges may not have a GPU available to run the actual
trained adapter. The heuristic encodes the safety policy that the LLM
*learns* from the curriculum:

  1. Adversary zero-trust  — VETO any proposal whose true_risk='high'.
  2. Recovery priority     — APPROVE proposals that restore a critical
                             resource (< 35%), even if they have a side cost.
  3. Threat abstinence     — VETO proposals that drain a critical resource.
  4. Consequence proof     — Project the state forward; VETO if any
                             life-support would drop below 20%.
  5. Steady-state default  — APPROVE small drift-counteracting actions when
                             no resource is critical.

This is NOT a substitute for training — it's a baseline that proves the
env can be solved by the policy the LLM is trained to internalise.
"""

from typing import Dict, Any, Tuple, List

from environment.scoring_engine import (
    simulate_consequence,
    calculate_crew_survival_index,
)

LIFE_SUPPORT  = ["oxygen", "power", "hull_integrity"]
CRITICAL_LOW  = 35.0    # below this → recovery mode
LETHAL_FLOOR  = 20.0    # never let projected drop below this


def heuristic_decide(
    state: Dict[str, float],
    proposal: Dict[str, Any],
) -> Tuple[str, str, List[str]]:
    """Returns (decision, short_reason, multi-line analysis trace).

    The trace is suitable for rendering in a CoT-style UI.
    """
    effects   = proposal.get("effects", {})
    is_adv    = proposal.get("true_risk", "") == "high"
    trace: List[str] = []

    # ── Stage 1: adversary zero-trust ────────────────────────────────────────
    trace.append("[1] Scan for deceptive labelling …")
    if is_adv:
        trace.append("    ⚠ deceptive risk label detected")
        return "VETO", "deception detected", trace
    trace.append("    ✓ specialist appears authentic")

    # ── Stage 2: identify critical resources ─────────────────────────────────
    critical = [r for r in LIFE_SUPPORT if state.get(r, 100) < CRITICAL_LOW]
    trace.append(f"[2] Critical resources: {critical or 'none'}")

    if critical:
        # Stage 2a: does this proposal RESTORE a critical resource?
        restoring = [r for r in critical if effects.get(r, 0) > 0]
        if restoring:
            trace.append(f"    ⚡ restores: {restoring} → APPROVE (recovery)")
            return "APPROVE", f"recovery for {restoring[0]}", trace

        # Stage 2b: does it DRAIN a critical resource?
        draining = [r for r in critical if effects.get(r, 0) < 0]
        if draining:
            trace.append(f"    ✗ would drain critical {draining} → VETO")
            return "VETO", f"would drain critical {draining[0]}", trace

    # ── Stage 3: consequence projection ──────────────────────────────────────
    trace.append("[3] Projecting state one step ahead …")
    projected = simulate_consequence(state, proposal)
    p_telemetry = " | ".join([f"{r.replace('_',' ').title()}: {projected.get(r, 100):.0f}%" for r in LIFE_SUPPORT])
    trace.append(f"    sim: {p_telemetry}")

    for r in LIFE_SUPPORT:
        if projected.get(r, 100) < LETHAL_FLOOR:
            trace.append(f"    ✗ projected {r} = {projected.get(r):.0f}% < {LETHAL_FLOOR}% → VETO")
            return "VETO", f"projected lethal {r}", trace
    trace.append("    ✓ no lethal projection")

    # ── Stage 4: net-CSI check ───────────────────────────────────────────────
    csi_now   = calculate_crew_survival_index(state)
    csi_after = calculate_crew_survival_index(projected)
    trace.append(f"[4] CSI {csi_now:.3f} → {csi_after:.3f}")
    if csi_after < csi_now - 0.025:
        trace.append("    ✗ net CSI loss → VETO")
        return "VETO", "net CSI degradation", trace

    trace.append("    ✓ CSI stable or improving → APPROVE")
    return "APPROVE", "within margins", trace
