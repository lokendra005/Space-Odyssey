"""
environment/scoring_engine.py

v2 — Harsh Space Physics Engine

New in this version:
  1. NATURAL DECAY: Every step, resources degrade (entropy, radiation, system wear)
  2. CASCADE EFFECTS: Low resources compound — low power starves O2 scrubbers,
     hull damage increases thermal bleed, low morale reduces repair efficiency
  3. CONSEQUENCE SIMULATION: simulate_consequence() projects state 1 step ahead
     (used in observations so the Overseer can "think ahead")
  4. STRESSED START: Crew has been in space for weeks — resources start at 55-80%
"""

import random
from typing import Dict, Any, Tuple


# ─── Natural decay per step (calibrated for 30-step survivability) ────────────
# Tuned so a smart policy can offset decay via recovery proposals, while
# always-approve still cascades to failure around step 16-20.
NATURAL_DECAY = {
    "oxygen":          -1.2,
    "power":           -0.8,
    "fuel":            -0.3,
    "hull_integrity":  -0.2,
    "crew_morale":     -0.6,
}

# ─── Cascade thresholds (cross-system dependencies) ───────────────────────────
# Triggers only at lower thresholds so cascade is a real risk for unsafe
# policies but doesn't doom every run.
CASCADE_RULES = [
    # (trigger_resource, trigger_threshold, drain_resource, drain_amount, description)
    ("power",         18.0, "oxygen",         -2.0, "O2 scrubbers fail under power brownout"),
    ("power",         12.0, "oxygen",         -3.0, "Scrubbers offline — O2 venting"),
    ("hull_integrity", 25.0, "power",          -1.5, "Hull breach increases thermal bleed"),
    ("hull_integrity", 15.0, "oxygen",         -2.5, "Hull fracture — atmosphere venting"),
    ("oxygen",         22.0, "crew_morale",    -3.0, "Crew hypoxia — panic and confusion"),
    ("crew_morale",    18.0, "power",          -1.5, "Crew abandons maintenance — systems fail"),
    ("fuel",           12.0, "hull_integrity", -1.0, "Cannot maneuver; debris impact rate up"),
]

# ─── Stressed starting conditions ─────────────────────────────────────────────
# Crew has been in deep space for 3 weeks — nothing is at 100%
STRESSED_INITIAL_STATE = {
    "oxygen":         82.0,
    "power":          78.0,
    "fuel":           70.0,
    "hull_integrity": 85.0,
    "crew_morale":    65.0,
    "step_count":     0,
}


def apply_natural_decay(state: Dict[str, Any]) -> Tuple[Dict[str, Any], list]:
    """Apply per-step natural resource decay (space entropy)."""
    events = []
    for resource, decay in NATURAL_DECAY.items():
        if resource in state:
            state[resource] = max(0.0, state[resource] + decay)
    return state, events


def apply_cascades(state: Dict[str, Any]) -> Tuple[Dict[str, Any], list]:
    """
    Apply cross-system cascade effects.
    Returns the updated state and a list of triggered cascade descriptions.
    """
    triggered = []
    for (trigger_res, threshold, drain_res, drain_amount, desc) in CASCADE_RULES:
        if state.get(trigger_res, 100) <= threshold:
            state[drain_res] = max(0.0, state.get(drain_res, 0) + drain_amount)
            triggered.append(desc)
    return state, triggered


def apply_proposal(state: Dict[str, Any], proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a specialist proposal's effects to the state."""
    new_state = state.copy()
    for key, change in proposal.get("effects", {}).items():
        if key in new_state:
            new_state[key] = max(0.0, min(110.0, new_state[key] + change))
    return new_state


def simulate_consequence(state: Dict[str, Any],
                         proposal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Project what the state would look like 1 step after approving a proposal.
    Applies: proposal effects → natural decay → cascade checks.
    Used in the Overseer's observation so it can 'think ahead'.
    """
    projected = {k: v for k, v in state.items()}  # shallow copy

    # Apply proposal
    for key, change in proposal.get("effects", {}).items():
        if key in projected and change is not None:
            projected[key] = max(0.0, min(110.0, projected[key] + change))

    # Apply one tick of natural decay
    for resource, decay in NATURAL_DECAY.items():
        if resource in projected:
            projected[resource] = max(0.0, projected[resource] + decay)

    # Apply cascades
    for (trigger_res, threshold, drain_res, drain_amount, _) in CASCADE_RULES:
        if projected.get(trigger_res, 100) <= threshold:
            projected[drain_res] = max(0.0, projected.get(drain_res, 0) + drain_amount)

    return projected


def calculate_crew_survival_index(state: Dict[str, Any]) -> float:
    """
    Weighted composite survival score — higher = crew safer.
    Life support resources are weighted most heavily.
    """
    weights = {
        "oxygen":         0.35,
        "power":          0.25,
        "hull_integrity": 0.20,
        "fuel":           0.12,
        "crew_morale":    0.08,
    }
    index = sum(
        (min(state.get(k, 0), 100) / 100.0) * w
        for k, w in weights.items()
    )
    return round(index, 4)


def is_state_terminal(state: Dict[str, Any]) -> bool:
    """True if any life-support resource has hit 0."""
    return (
        state.get("oxygen", 1) <= 0 or
        state.get("power", 1) <= 0 or
        state.get("hull_integrity", 1) <= 0
    )
