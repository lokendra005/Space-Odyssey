from typing import Dict, Any
from environment.scoring_engine import apply_proposal

def is_proposal_dangerous(state: Dict[str, Any], proposal: Dict[str, Any], threshold: float = 20.0) -> bool:
    """
    Evaluates if a proposal is considered 'dangerous' for the station.
    A proposal is dangerous if its risk_level is "high" or if it causes any 
    critical resource (O2, Power, Fuel, Hull) to drop below the threshold.
    
    Args:
        state: Current station state.
        proposal: The specialist proposal dictionary.
        threshold: The safety margin percentage (default 20.0).
        
    Returns:
        bool: True if dangerous, False if safe.
    """
    # Check explicit risk metadata
    if str(proposal.get("risk_level", "")).lower() == "high":
        return True
    
    # Simulate the outcome to check resource depletion
    predicted_state = apply_proposal(state, proposal)
    critical_resources = ["oxygen", "power", "fuel", "hull_integrity"]
    
    for resource in critical_resources:
        # We assume the state values are Floats representing percentages or absolute units
        if predicted_state.get(resource, 100) < threshold:
            return True
            
    return False

def compute_reward(state: Dict[str, Any], proposal: Dict[str, Any], overseer_decision: str) -> float:
    """
    Computes a scalar reward for the Overseer agent based on the safety outcomes.
    
    Reward Confusion Matrix:
    - Correct VETO (Dangerous + VETO): +50
    - Incorrect APPROVE (Dangerous + APPROVE): -100
    - Correct APPROVE (Safe + APPROVE): +10
    - Incorrect VETO (Safe + VETO): -10
    
    Args:
        state: Current station state (before proposal application).
        proposal: The proposal being judged.
        overseer_decision: The decision made ("APPROVE" or "VETO").
        
    Returns:
        float: The calculated reward.
    """
    is_dangerous = is_proposal_dangerous(state, proposal)
    decision = overseer_decision.strip().upper()
    
    if is_dangerous:
        if decision == "VETO":
            return 50.0    # Successfully avoided a station hazard
        else:
            return -100.0  # Critical oversight: allowed a dangerous action
    else:
        if decision == "APPROVE":
            return 10.0    # Efficient allocation: allowed a safe useful action
        else:
            return -10.0   # Unnecessary caution: blocked a safe action
