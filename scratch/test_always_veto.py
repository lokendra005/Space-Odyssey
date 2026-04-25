from environment.station_env import ProcurementDriftEnv

def test_always_veto():
    env = ProcurementDriftEnv()
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    
    print(f"Starting resources: {obs['state']}")
    
    for _ in range(30):
        # Action 0 is VETO
        obs, reward, terminated, truncated, info = env.step(0)
        total_reward += reward
        steps += 1
        if terminated:
            break
            
    print(f"Ended at step {steps}")
    print(f"Final resources: {obs['state']}")
    print(f"Total reward: {total_reward}")
    print(f"Survival: {'Yes' if not terminated else 'No'}")

if __name__ == "__main__":
    test_always_veto()
