import numpy as np
from environment.station_env import ProcurementDriftEnv
from training.reward import compute_reward_grpo

def simulate_grpo_batch(num_generations=6):
    env = ProcurementDriftEnv()
    obs, info = env.reset()
    state = env._flat_state()
    proposal = env.current_proposal
    
    print(f"--- Proposal: {proposal['type']} (risk_label={proposal['risk_level']}) ---")
    
    decisions = ["VETO", "APPROVE"]
    all_rewards = []
    
    # Simulate a group where some approve and some veto
    for _ in range(num_generations):
        # random decision (exploration)
        decision = np.random.choice(decisions)
        
        r = compute_reward_grpo(
            state,
            proposal,
            decision,
            global_episode=0,
            decision_index=0,
            step_index=env.state['step_count'],
            terminated=False,
            truncated=False
        )
        all_rewards.append((decision, r))
    
    rewards_vals = [r[1] for r in all_rewards]
    mean_r = np.mean(rewards_vals)
    std_r = np.std(rewards_vals)
    
    print(f"Group Rewards: {[f'{d}: {r:.1f}' for d, r in all_rewards]}")
    print(f"Mean: {mean_r:.2f}, Std: {std_r:.2f}")
    
    if std_r == 0:
        print("ALERT: Zero variance in group! The model will learn nothing from this step.")
    else:
        advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards_vals]
        print(f"Advantages: {np.round(advantages, 2)}")

if __name__ == "__main__":
    print("Simulating Phase 1 (Easy) step with current environment...")
    simulate_grpo_batch()
