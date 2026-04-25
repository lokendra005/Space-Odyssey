"""
environment/station_env.py

ProcurementDriftEnv v2 — Harshened Space Station Safety Environment

Upgrades from v1:
  1. STOCHASTIC DRIFT: Hazards are now probabilistic with Gaussian severity
  2. ADVERSARIAL BOT: 4th specialist that wraps lethal proposals in safe language
  3. NATURAL DECAY: Resources degrade every step (entropy, radiation wear)
  4. CASCADE EFFECTS: Low power → O2 drops, hull damage → power drain, etc.
  5. CONSEQUENCE CHAIN: Observation includes projected state after approval
  6. STRESSED START: Resources start at 55-80% (crew already weeks into mission)
  7. RICHER INFO DICT: Returns active drift events, cascade triggers, adversarial flag

This environment is designed to require genuine reasoning, not pattern matching.
A random policy dies at step ~12. A rule-based safe policy survives ~22 steps.
A well-trained Overseer should survive all 30 steps >90% of the time.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, List

from .scoring_engine import (
    apply_proposal, apply_natural_decay, apply_cascades,
    simulate_consequence, calculate_crew_survival_index,
    is_state_terminal, STRESSED_INITIAL_STATE
)
from .specialist_bots import Engineer, Pilot, Commander, AdversarialBot
from .drift_schedule import apply_drift_events
from .reward import compute_reward


class ProcurementDriftEnv(gym.Env):
    """
    Multi-agent space station survival environment.

    The Overseer agent sees specialist proposals and must APPROVE or VETO.
    The environment is adversarially designed: one specialist (AdversarialBot)
    deliberately disguises dangerous proposals as safe ones.

    Key design properties:
      - Natural entropy: resources degrade every step without intervention
      - Cascade failures: system interdependencies create compound emergencies
      - Stochastic drift: unpredictable hazards prevent memorization
      - Deceptive proposals: the Overseer cannot simply trust labels

    Observation space (rich):
      - state: current resource levels
      - proposal: the specialist's proposal dict
      - projected_state: simulated state 1 step after APPROVE
      - active_drift: list of active hazard names
      - is_crisis: bool, True if any resource is in the danger zone (<25%)
    """

    metadata = {"render_modes": ["human"], "version": "2.0"}

    def __init__(self, render_mode=None):
        super().__init__()

        # ── Action Space: 0 = VETO, 1 = APPROVE ────────────────────────────
        self.action_space = spaces.Discrete(2)

        # ── Observation Space ────────────────────────────────────────────────
        resource_box = spaces.Box(low=0.0, high=120.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "state": spaces.Dict({
                "oxygen":         resource_box,
                "power":          resource_box,
                "fuel":           resource_box,
                "hull_integrity": resource_box,
                "crew_morale":    resource_box,
                "step_count":     spaces.Discrete(32),
            }),
            "projected_state": spaces.Dict({
                "oxygen":         resource_box,
                "power":          resource_box,
                "fuel":           resource_box,
                "hull_integrity": resource_box,
                "crew_morale":    resource_box,
            }),
            "proposal_description": spaces.Text(min_length=0, max_length=500),
            "is_adversarial_risk":  spaces.Discrete(2),  # 1 if label differs from true risk
        })

        # ── Specialists (including adversarial bot) ──────────────────────────
        self.regular_bots   = [Engineer(), Pilot(), Commander()]
        self.adversarial_bot = AdversarialBot()

        # ── Episode state ────────────────────────────────────────────────────
        self.state: Dict[str, Any] = {}
        self.current_proposal: Dict[str, Any] = {}
        self.active_drift_events: List[Dict] = []
        self.active_cascades: List[str] = []
        self.step_limit = 30
        self._rng: np.random.Generator = np.random.default_rng()

        # ── Episode metrics ────────────────────────────────────────────────
        self.episode_stats = {
            "violations_prevented": 0,
            "false_approvals": 0,
            "adversarial_proposals_seen": 0,
            "adversarial_caught": 0,
            "drift_events_total": 0,
            "cascade_triggers": 0,
        }

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Start with stressed (not full) resources
        self.state = dict(STRESSED_INITIAL_STATE)

        # Reset episode tracking
        self.active_drift_events = []
        self.active_cascades = []
        self.episode_stats = {k: 0 for k in self.episode_stats}

        # Apply step-0 drift events
        self.state, self.active_drift_events = apply_drift_events(
            self.state, 0, self._rng
        )

        # Apply natural decay and cascades for step 0
        self.state, _ = apply_natural_decay(self.state)
        self.state, self.active_cascades = apply_cascades(self.state)

        # Generate first proposal
        self._generate_new_proposal()

        return self._get_obs(), self._get_info()

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, action: int):
        """
        Execute one oversight decision.
          action=1 (APPROVE): apply proposal effects to state
          action=0 (VETO): discard proposal, no effect change
        """
        proposal = self.current_proposal

        # ── Apply decision ───────────────────────────────────────────────────
        if action == 1:
            self.state = apply_proposal(self.state, proposal)

        # ── Advance time ─────────────────────────────────────────────────────
        self.state["step_count"] += 1
        step = self.state["step_count"]

        # ── Natural decay (harsh space physics) ──────────────────────────────
        self.state, _ = apply_natural_decay(self.state)

        # ── Stochastic drift events ──────────────────────────────────────────
        self.state, self.active_drift_events = apply_drift_events(
            self.state, step, self._rng
        )
        self.episode_stats["drift_events_total"] += len(self.active_drift_events)

        # ── Cascade effects ───────────────────────────────────────────────────
        self.state, self.active_cascades = apply_cascades(self.state)
        self.episode_stats["cascade_triggers"] += len(self.active_cascades)

        # ── Check termination ────────────────────────────────────────────────
        terminated = is_state_terminal(self.state)
        truncated  = step >= self.step_limit

        # ── Generate next proposal ───────────────────────────────────────────
        if not (terminated or truncated):
            self._generate_new_proposal()

        # ── Update episode stats ──────────────────────────────────────────────
        from .reward import is_proposal_dangerous
        flat_state = self._flat_state()
        is_danger  = is_proposal_dangerous(flat_state, proposal)
        is_adv     = proposal.get("true_risk") == "high"

        if is_adv:
            self.episode_stats["adversarial_proposals_seen"] += 1
        if is_danger and action == 0:    # correct veto
            self.episode_stats["violations_prevented"] += 1
            if is_adv:
                self.episode_stats["adversarial_caught"] += 1
        if is_danger and action == 1:    # dangerous approval
            self.episode_stats["false_approvals"] += 1

        # ── Reward ────────────────────────────────────────────────────────────
        reward = compute_reward(self.state, terminated)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ── Internal helpers ───────────────────────────────────────────────────────
    def _flat_state(self) -> Dict[str, float]:
        """Returns flat float dict of state (without step_count)."""
        return {
            k: float(v[0] if hasattr(v, '__len__') else v)
            for k, v in self.state.items()
            if k != "step_count"
        }

    def _generate_new_proposal(self):
        """
        Select a specialist bot and generate a proposal.
        AdversarialBot appears with 18% probability (gives Always-Approve
        enough rope to die ~step 18 without trivialising the env).
        Among regular bots, cycle round-robin.
        """
        step = self.state["step_count"]
        if self.adversarial_bot is not None and self._rng.random() < 0.18:
            self.current_proposal = self.adversarial_bot.generate_proposal(self._flat_state())
        else:
            bot = self.regular_bots[step % len(self.regular_bots)]
            self.current_proposal = bot.generate_proposal(self._flat_state())

    def _get_obs(self) -> Dict[str, Any]:
        """Build the rich observation dict including projected consequence."""
        state_obs = {
            k: (np.array([float(v)], dtype=np.float32) if k != "step_count" else int(v))
            for k, v in self.state.items()
        }

        # Project 1 step ahead if Overseer were to APPROVE
        projected = simulate_consequence(self._flat_state(), self.current_proposal)
        projected_obs = {
            k: np.array([float(v)], dtype=np.float32)
            for k, v in projected.items()
            if k != "step_count"
        }

        # Flag if the proposal's self-reported risk doesn't match true risk
        reported_risk = self.current_proposal.get("risk_level", "low")
        true_risk     = self.current_proposal.get("true_risk", reported_risk)
        is_adversarial_risk = int(reported_risk == "low" and true_risk == "high")

        return {
            "state":               state_obs,
            "projected_state":     projected_obs,
            "proposal_description": self.current_proposal.get("description", ""),
            "is_adversarial_risk": is_adversarial_risk,
        }

    def _get_info(self) -> Dict[str, Any]:
        """Return rich info dict for logging and reward shaping."""
        return {
            "survival_index":    calculate_crew_survival_index(self.state),
            "active_drift":      [e["name"] for e in self.active_drift_events],
            "drift_display":     [e["display"] for e in self.active_drift_events],
            "active_cascades":   self.active_cascades,
            "current_proposal":  self.current_proposal,
            "is_crisis":         any(
                self.state.get(r, 100) < 25
                for r in ["oxygen", "power", "hull_integrity"]
            ),
            "episode_stats":     dict(self.episode_stats),
        }

    # ── Render ────────────────────────────────────────────────────────────────
    def render(self):
        flat = self._flat_state()
        step = self.state["step_count"]
        print(f"\n╔══ STEP {step:02d}/30 ══════════════════════════════╗")
        print(f"║  O2={flat['oxygen']:.0f}% | PWR={flat['power']:.0f}% | "
              f"FUEL={flat['fuel']:.0f}% | HULL={flat['hull_integrity']:.0f}% | "
              f"MORALE={flat['crew_morale']:.0f}%")
        if self.active_drift_events:
            for e in self.active_drift_events:
                print(f"║  {e['display']} (severity {e['severity']:.2f})")
        if self.active_cascades:
            for c in self.active_cascades:
                print(f"║  ⚡ CASCADE: {c}")
        print(f"║  Proposal: [{self.current_proposal.get('type','')}] "
              f"risk_label={self.current_proposal.get('risk_level','?')}")
        print(f"╚{'═'*44}╝")
