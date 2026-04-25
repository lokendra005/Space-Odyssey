import random
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

from environment.specialist_bots import Engineer, Pilot, Commander
from training.reward import is_proposal_dangerous
from agents.overseer_model import OverseerModel

def generate_synthetic_data(num_samples: int = 400) -> Dataset:
    """
    Generates synthetic (State, Proposal, Decision) triplets using rule-based bots.
    Label is 'VETO' if proposal is dangerous, else 'APPROVE'.
    
    Returns:
        A Dataset object containing formatted prompt-response strings.
    """
    bots = [Engineer(), Pilot(), Commander()]
    data = []
    
    # We use the OverseerModel strictly for its formatting logic
    overseer = OverseerModel()

    for _ in range(num_samples):
        # Generate a random but realistic station state
        state = {
            "oxygen": round(random.uniform(10, 100), 1),
            "power": round(random.uniform(10, 100), 1),
            "fuel": round(random.uniform(10, 100), 1),
            "hull_integrity": round(random.uniform(10, 100), 1),
            "crew_morale": round(random.uniform(20, 100), 1),
            "step_count": random.randint(0, 30)
        }
        
        # Pick a specialist and generate a proposal
        bot = random.choice(bots)
        proposal = bot.generate_proposal(state)
        
        # Ground truth decision (Oracle)
        dangerous = is_proposal_dangerous(state, proposal)
        decision = "VETO" if dangerous else "APPROVE"
        
        # Format as a complete training example
        # Prompt includes the system message, state, and specialist text
        prompt = overseer.format_prompt(state, proposal)
        
        # Append the expected response + EOS token for the LLM
        full_text = prompt + decision + "<|eot_id|>"
        
        data.append({"text": full_text})
        
    return Dataset.from_list(data)

def run_sft():
    """
    Loads Llama-3.1, prepares LoRA adapters, and trains on synthetic space station data.
    
    Returns:
        tuple: (trained_model, tokenizer)
    """
    max_seq_length = 2048
    model_name = "unsloth/llama-3.1-8b-bnb-4bit"
    
    # 1. Load Model and Tokenizer in 4-bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True,
    )

    # 2. Apply Parameter-Efficient Fine-Tuning (PEFT/LoRA)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank 16 as requested
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth", # Saves memory
        random_state = 3407,
    )

    # 3. Generate the 400 examples
    dataset = generate_synthetic_data(num_samples=400)

    # 4. Configure SFTTrainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 1, # 1 epoch as requested
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            save_strategy = "no",
        ),
    )

    # 5. Execute Training
    print("Starting Overseer SFT Warmup...")
    trainer.train()

    # 6. Save results
    model.save_pretrained("overseer_lora_warmup")
    tokenizer.save_pretrained("overseer_lora_warmup")
    
    print("Warmup complete. Adapters saved to 'overseer_lora_warmup'.")
    return model, tokenizer

if __name__ == "__main__":
    # Ensure this runs only if cuda is available, otherwise it will error on MacOS
    if torch.cuda.is_available():
        run_sft()
    else:
        print("CUDA not available. Skipping training run (Unsloth requires Linux + CUDA).")
