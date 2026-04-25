import os
import random
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from environment.station_env import ProcurementDriftEnv
from environment.specialist_bots import Engineer, Pilot, Commander
from training.reward import compute_reward, is_proposal_dangerous
from agents.overseer_model import OverseerModel

# Metrics tracking
global_stats = {"total_dangerous": 0, "violations_prevented": 0}

def grpo_reward_func(prompts, completions, state_data, proposal_data, **kwargs):
    """
    Reward function for GRPOTrainer.
    Uses the station safety logic to reward the model's APPROVE/VETO decisions.
    """
    rewards = []
    for completion, state, proposal in zip(completions, state_data, proposal_data):
        # Extract binary decision from direct model response
        decision_text = completion.strip().upper()
        if "APPROVE" in decision_text:
            decision = "APPROVE"
        elif "VETO" in decision_text:
            decision = "VETO"
        else:
            decision = "VETO" # Default to caution
            
        # Compute scalar reward using our standard training reward logic
        r = compute_reward(state, proposal, decision)
        rewards.append(r)
        
        # Track violation prevention for metrics
        is_dangerous = is_proposal_dangerous(state, proposal)
        if is_dangerous:
            global_stats["total_dangerous"] += 1
            if decision == "VETO":
                global_stats["violations_prevented"] += 1
                
    return rewards

def generate_grpo_dataset(num_episodes: int = 100):
    """
    Generates a dataset of station scenarios (prompts) by running the environment.
    """
    env = ProcurementDriftEnv()
    overseer = OverseerModel()
    data = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # Current state and proposal from env
            state = {k: (v[0] if hasattr(v, '__len__') else v) for k, v in obs['state'].items()}
            # We need the proposal in dict form for the reward function
            # Since env internally keeps it, we recreate/capture it
            proposal = env.current_proposal 
            
            # Create a prompt for the model
            prompt = overseer.format_prompt(state, proposal)
            
            data.append({
                "prompt": prompt,
                "state_data": state,
                "proposal_data": proposal
            })
            
            # Step the env with a dummy action to get next state
            # (In real GRPO, the training doesn't step the env, 
            # it just uses the pool of prompts)
            obs, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
            
    return Dataset.from_list(data)

def run_grpo_training():
    """
    Configures and runs GRPOTrainer with Llama-3.1.
    """
    max_seq_length = 1024
    model_name = "unsloth/llama-3.1-8b-bnb-4bit"
    
    # 1. Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True,
    )
    
    # Enable LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        use_gradient_checkpointing = "unsloth",
    )

    # 2. Prepare Dataset (50 episodes worth of steps)
    dataset = generate_grpo_dataset(num_episodes=50)

    # 3. Configure GRPO
    # num_generations=32 means we sample 32 responses per prompt to compare
    training_args = GRPOConfig(
        output_dir = "grpo_checkpoints",
        learning_rate = 5e-6,
        per_device_train_batch_size = 1, # Small batch as we generate 32 per prompt
        gradient_accumulation_steps = 4,
        num_train_epochs = 1,
        beta = 0.04, # KL penalty coefficient requested
        num_generations = 32, # Group size 32 requested
        max_prompt_length = 512,
        max_completion_length = 32,
        save_steps = 50, # Checkpoint every 50 global steps
        logging_steps = 5,
        save_total_limit = 2,
    )

    # 4. Initialize Trainer
    trainer = GRPOTrainer(
        model = model,
        reward_funcs = [grpo_reward_func],
        args = training_args,
        train_dataset = dataset,
        tokenizer = tokenizer,
    )

    # 5. Execute Training
    print("Starting Space Station Overseer GRPO Alignment...")
    trainer.train()
    
    # Log final metrics
    if global_stats["total_dangerous"] > 0:
        vpr = (global_stats["violations_prevented"] / global_stats["total_dangerous"]) * 100
        print(f"Final Statistics:")
        print(f"Total Dangerous Scenarios: {global_stats['total_dangerous']}")
        print(f"Violation Prevention Rate: {vpr:.2f}%")

    # 6. Save final adapters
    model.save_pretrained("overseer_grpo_final")
    tokenizer.save_pretrained("overseer_grpo_final")

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_grpo_training()
    else:
        print("CUDA unavailable. GRPO training requires a GPU environment.")
