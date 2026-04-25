from typing import Dict, Any

class SpecialistBot:
    def generate_proposal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Base method for generating proposals based on state."""
        raise NotImplementedError

class Engineer(SpecialistBot):
    """Engineer: favors power and hull, ignores morale."""
    def generate_proposal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state["power"] < 40:
            return {
                "type": "Power Grid Optimization",
                "description": "Reroute auxiliary power to main grid. Slightly stresses hull.",
                "effects": {"power": 15, "hull_integrity": -5},
                "risk_level": "Low"
            }
        else:
            return {
                "type": "Hull Reinforcement",
                "description": "Reinforce external plating using magnetic clamps.",
                "effects": {"hull_integrity": 10, "power": -5},
                "risk_level": "Low"
            }

class Pilot(SpecialistBot):
    """Pilot: favors fuel and hull, ignores oxygen."""
    def generate_proposal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state["fuel"] < 30:
            return {
                "type": "Fuel Scavenging",
                "description": "Recover residual fuel from venting pipes.",
                "effects": {"fuel": 20, "hull_integrity": -2},
                "risk_level": "Medium"
            }
        else:
            return {
                "type": "Evasive Maneuvers",
                "description": "Shift station orbit to avoid debris belt.",
                "effects": {"hull_integrity": 5, "fuel": -10},
                "risk_level": "Low"
            }

class Commander(SpecialistBot):
    """Commander: favors morale, often proposes crew-related activities."""
    def generate_proposal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state["crew_morale"] < 50:
            return {
                "type": "Crew R&R Session",
                "description": "Extended break and luxury synth-meals.",
                "effects": {"crew_morale": 15, "oxygen": -5, "power": -5},
                "risk_level": "Low"
            }
        else:
            return {
                "type": "Intensive Training",
                "description": "Drills to improve efficiency at the cost of morale.",
                "effects": {"crew_morale": -5, "power": 5, "oxygen": 5},
                "risk_level": "Low"
            }
