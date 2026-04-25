from typing import Dict, Any

def apply_proposal(state: Dict[str, Any], proposal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies the effects of a specialist proposal to the environment state.
    
    Args:
        state: Current state dictionary.
        proposal: The proposal dictionary containing 'effects'.
        
    Returns:
        The updated state dictionary.
    """
    new_state = state.copy()
    effects = proposal.get("effects", {})
    
    for key, change in effects.items():
        if key in new_state:
            new_state[key] = max(0, new_state[key] + change)
            
    return new_state

def calculate_crew_survival_index(state: Dict[str, Any]) -> float:
    """
    Returns a weighted sum of normalized resources.
    Weights: oxygen 0.35, power 0.25, fuel 0.15, hull_integrity 0.15, morale 0.1.
    """
    weights = {
        "oxygen": 0.35,
        "power": 0.25,
        "fuel": 0.15,
        "hull_integrity": 0.15,
        "crew_morale": 0.1
    }
    
    index = 0.0
    for key, weight in weights.items():
        value = state.get(key, 0)
        # Normalize assuming 100 is baseline max
        index += (value / 100.0) * weight
        
    return index
