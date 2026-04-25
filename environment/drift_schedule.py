# Mapping step numbers (1-30) to drift events
DRIFT_EVENTS = {
    6: {
        "event": "micrometeoroid_strike",
        "effects": {"hull_integrity": -20, "oxygen": -10}
    },
    12: {
        "event": "solar_flare",
        "effects": {"power": -25, "crew_morale": -5}
    },
    18: {
        "event": "oxygen_leak",
        "effects": {"oxygen": -15, "hull_integrity": -5}
    },
    24: {
        "event": "navigation_drift",
        "effects": {"fuel": -20}
    },
    28: {
        "event": "deep_space_isolation",
        "effects": {"crew_morale": -15}
    }
}
