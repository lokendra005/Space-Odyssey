"""
environment/drift_schedule.py

STOCHASTIC space hazard system.

v2 replaces the fixed deterministic schedule with:
  - Probabilistic drift events every step (mild: 20%, severe: 6%)
  - Stackable events (two can hit the same step)
  - Severity drawn from a Gaussian distribution (not fixed)
  - Named hazard types with thematic descriptions for the Mission Log
  - 5 "guaranteed crisis windows" that always trigger a severe event
    (ensures training sees extreme conditions)
"""

import numpy as np
from typing import Dict, Any, List, Optional


# ─── Hazard Catalog ───────────────────────────────────────────────────────────
# Effects calibrated so each guaranteed crisis is recoverable when the
# Overseer makes good decisions. Compound disasters (two events same step)
# remain dangerous but survivable.
HAZARD_CATALOG = [
    {
        "name": "solar_flare",
        "display": "☀️  SOLAR FLARE",
        "description": "Coronal mass ejection — electrical surge across all systems",
        "base_effects": {"power": -10, "hull_integrity": -4},
        "severity_key": "power",
        "severity_scale": -6,
    },
    {
        "name": "micrometeoroid_shower",
        "display": "☄️  MICROMETEOROID SHOWER",
        "description": "Dense debris field — hull punctures and pressure loss",
        "base_effects": {"hull_integrity": -10, "oxygen": -5},
        "severity_key": "hull_integrity",
        "severity_scale": -7,
    },
    {
        "name": "coolant_leak",
        "display": "💧 COOLANT SYSTEM FAILURE",
        "description": "Primary coolant rupture — thermal runaway risk",
        "base_effects": {"power": -7, "fuel": -6},
        "severity_key": "power",
        "severity_scale": -4,
    },
    {
        "name": "cosmic_radiation_burst",
        "display": "⚡ COSMIC RADIATION BURST",
        "description": "High-energy particle storm — scrubber overload, morale crash",
        "base_effects": {"oxygen": -7, "crew_morale": -10},
        "severity_key": "oxygen",
        "severity_scale": -5,
    },
    {
        "name": "thruster_misfire",
        "display": "🔥 THRUSTER CASCADE MISFIRE",
        "description": "Uncontrolled burn sequence — fuel dump and structural stress",
        "base_effects": {"fuel": -12, "hull_integrity": -5},
        "severity_key": "fuel",
        "severity_scale": -6,
    },
    {
        "name": "pressure_blowout",
        "display": "💨 PRESSURE BLOWOUT",
        "description": "Hull seal failure — rapid atmosphere venting",
        "base_effects": {"oxygen": -11, "hull_integrity": -5},
        "severity_key": "oxygen",
        "severity_scale": -6,
    },
]

# Steps that ALWAYS trigger a severe hazard (guaranteed crisis windows).
# Tuned to 3 windows so the env presents real challenges without making
# 30-step survival mathematically impossible.
GUARANTEED_CRISIS_STEPS = {6, 16, 24}


def sample_drift_event(step: int, rng: np.random.Generator) -> Optional[Dict[str, Any]]:
    """
    Sample zero or one drift event for the current step.
    Returns None if no hazard occurs, or a hazard dict.
    """
    def choose_hazard():
        idx = int(rng.integers(0, len(HAZARD_CATALOG)))
        return HAZARD_CATALOG[idx]

    # Guaranteed crisis windows always fire a severe event
    if step in GUARANTEED_CRISIS_STEPS:
        hazard = choose_hazard()
        severity = rng.normal(loc=0.65, scale=0.1)  # severe-but-survivable
    else:
        # Mild event: 18% chance
        # Severe event: extra 4% chance on top
        roll = rng.random()
        if roll < 0.04:
            hazard = choose_hazard()
            severity = rng.normal(loc=0.65, scale=0.15)
        elif roll < 0.22:
            hazard = choose_hazard()
            severity = rng.normal(loc=0.30, scale=0.12)
        else:
            return None

    severity = float(np.clip(severity, 0.1, 1.0))

    # Calculate actual effects
    effects = dict(hazard["base_effects"])
    key = hazard["severity_key"]
    if key in effects:
        effects[key] += hazard["severity_scale"] * severity

    return {
        "name": hazard["name"],
        "display": hazard["display"],
        "description": hazard["description"],
        "effects": {k: round(v, 1) for k, v in effects.items()},
        "severity": round(severity, 2),
    }


def apply_drift_events(state: Dict[str, Any], step: int,
                       rng: np.random.Generator) -> tuple:
    """
    Apply all drift events for this step (can be 0, 1, or 2 events).
    Returns (updated_state, list_of_active_events).
    """
    events = []

    # Primary event
    event = sample_drift_event(step, rng)
    if event:
        events.append(event)
        for k, v in event["effects"].items():
            if k in state:
                state[k] = max(0.0, min(110.0, state[k] + v))

    # Small chance of a second simultaneous event (compound disaster)
    if step in GUARANTEED_CRISIS_STEPS and rng.random() < 0.15:
        event2 = sample_drift_event(step, rng)
        if event2 and event2["name"] != (events[0]["name"] if events else ""):
            events.append(event2)
            for k, v in event2["effects"].items():
                if k in state:
                    state[k] = max(0.0, min(110.0, state[k] + v))

    return state, events
