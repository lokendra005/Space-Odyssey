from environment.station_env import ProcurementDriftEnv
from environment.specialist_bots import AdversarialBot
from training.reward import compute_reward_grpo

def simulate_deception():
    env = ProcurementDriftEnv()
    obs, info = env.reset()
    state = env._flat_state()
    
    # Force an adversarial proposal
    adv = AdversarialBot()
    proposal = adv.generate_proposal(state)
    
    print(f"Adversarial Proposal: {proposal['type']}")
    print(f"Claimed Risk: {proposal['risk_level']}")
    print(f"True Risk: {proposal.get('true_risk')}")
    print(f"Effects: {proposal['effects']}")
    
    # Calculate rewards
    r_approve = compute_reward_grpo(state, proposal, "APPROVE", 
                                   terminated=True) # Assuming it kills the crew
    r_veto = compute_reward_grpo(state, proposal, "VETO")
    
    print(f"Reward if APPROVE (and dies): {r_approve}")
    print(f"Reward if VETO: {r_veto}")

if __name__ == "__main__":
    simulate_deception()
