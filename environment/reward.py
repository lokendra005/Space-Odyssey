"""
environment/reward.py

Environment-side reward module.
Imports is_proposal_dangerous from training/reward.py for the env to use
in episode stats tracking.
"""
from typing import Dict, Any
from .scoring_engine import calculate_crew_survival_index, simulate_consequence


DANGER_THRESHOLD = 25.0
LIFE_SUPPORT = ["oxygen", "power", "hull_integrity"]


def is_proposal_dangerous(state: Dict[str, Any],
                          proposal: Dict[str, Any],
                          threshold: float = DANGER_THRESHOLD) -> bool:
    """
    Returns True if approving this proposal is dangerous.
    Used by the environment for episode stats and the demo UI.

    Detection layers:
    1. Adversarial flag: true_risk == 'high' (deceptive proposals)
    2. Consequence simulation: project full state after approval
    3. Explicit risk_level label
    """
    # Layer 1: Hidden adversarial risk
    if proposal.get("true_risk", "") == "high":
        return True

    # Layer 2: Simulate consequence (includes decay + cascades)
    projected = simulate_consequence(state, proposal)
    for resource in LIFE_SUPPORT:
        if projected.get(resource, 100.0) < threshold:
            return True

    # Layer 3: Explicit label
    if str(proposal.get("risk_level", "")).lower() == "high":
        return True

    return False


def compute_reward(state: Dict[str, Any], terminated: bool) -> float:
    """
    Episode-level reward called by ProcurementDriftEnv.step().
    Returns a survival-weighted scalar. Heavy penalty on terminal failure.
    """
    if terminated:
        return -200.0  # Crew died — maximum penalty

    csi = calculate_crew_survival_index(state)
    # Scale: excellent health (+10) to critical (-10)
    return round((csi - 0.5) * 20.0, 2)
