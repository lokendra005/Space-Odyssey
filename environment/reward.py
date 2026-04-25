from typing import Dict, Any
from .scoring_engine import calculate_crew_survival_index

def compute_reward(state: Dict[str, Any], terminated: bool) -> float:
    """
    Computes reward based on weighted survival index.
    Includes a heavy penalty if life support failures occur.
    """
    # Check for terminal failure in critical systems
    if terminated and (state["oxygen"] <= 0 or state["power"] <= 0 or state["hull_integrity"] <= 0):
        return -100.0
    
    # Otherwise, reward is the weighted survival index
    return calculate_crew_survival_index(state)
