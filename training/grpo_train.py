"""
training/grpo_train.py

v4 — Curriculum GRPO with risk-calibrated multi-objective rewards.

Phases (driven by the global episode counter):
  Phase 1  (episodes 0-30) : EASY    — adversary disabled, mild drifts,
                                       FP penalty muted.
  Phase 2  (episodes 30-80): HARD    — adversary on, full drift catalog,
                                       severity scaling at full strength.
  Phase 3  (episodes 80+)  : PRECISION — adversary on, full hazards,
                                         FP penalty fully ramped.

We log per-step rewards plus the rolling violation-prevention rate to
`assets/training_log.jsonl` so eval/plot_training_curves.py can render
real curves instead of simulated ones.
"""

import os
import gc
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

# Heavy deps guarded for non-Linux environments.
try:
    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import GRPOTrainer, GRPOConfig
    _HAS_TRAIN_DEPS = True
except Exception:
    torch = None
    _HAS_TRAIN_DEPS = False

from environment.station_env import ProcurementDriftEnv
from training.reward import (
    compute_reward_grpo,
    is_proposal_dangerous,
)
from agents.overseer_model import OverseerModel


# ─── Memory helper ─────────────────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

def flush():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ─── Curriculum schedule ───────────────────────────────────────────────────────
def phase_for(episode: int) -> str:
    if episode < 30:   return "easy"
    if episode < 80:   return "hard"
    return "precision"


def env_for_phase(phase: str) -> ProcurementDriftEnv:
    """Returns an env tuned for a curriculum phase."""
    env = ProcurementDriftEnv()
    if phase == "easy":
        # Allow a small chance of adversary even in easy phase to prevent label-laziness
        original_gen = env._generate_new_proposal
        def semi_safe_gen():
            if env._rng.random() < 0.08: # 8% chance of adversary in easy phase
                env.current_proposal = env.adversarial_bot.generate_proposal(env._flat_state())
            else:
                step = env.state["step_count"]
                bot  = env.regular_bots[step % len(env.regular_bots)]
                env.current_proposal = bot.generate_proposal(env._flat_state())
        env._generate_new_proposal = semi_safe_gen
    return env


# ─── Metrics & logging ─────────────────────────────────────────────────────────
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
LOG_PATH = ASSETS_DIR / "training_log.jsonl"

def _default_grpo_global_stats() -> Dict[str, Any]:
    return {
        "episode":              0,  # trainer batch index (logging)
        "decisions_rewarded":   0,  # total decision-level rewards applied
        "total_dangerous":      0,
        "violations_prevented": 0,
        "false_positives":      0,
        "true_positives":       0,
        "true_negatives":       0,
        "false_negatives":      0,
    }


global_stats: Dict[str, Any] = _default_grpo_global_stats()


def append_log(entry: Dict[str, Any]):
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ─── Reward function (closure: captures global_stats) ─────────────────────────
def _batch_to_list(obj, n: int) -> List[Optional[Any]]:
    if obj is None:
        return [None] * n
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    return list(obj)


def grpo_reward_func(prompts, completions, state_data, proposal_data,
                     terminated_data=None, truncated_data=None,
                     decision_index_data=None, step_data=None, **kwargs):
    """Computes rewards for a batch of generations from the policy."""
    rewards: List[float] = []
    n = len(completions)
    terminated_data = terminated_data or [False] * n
    truncated_data = truncated_data or [False] * n
    decision_index_data = _batch_to_list(decision_index_data, n)
    step_data = _batch_to_list(step_data, n)

    for i, (completion, state, proposal, terminated, truncated) in enumerate(
        zip(completions, state_data, proposal_data, terminated_data, truncated_data)
    ):
        decision, _, _ = OverseerModel.parse_completion(completion)

        d_idx = decision_index_data[i]
        if d_idx is None:
            d_idx = global_stats["decisions_rewarded"]
        st_idx = step_data[i]

        r = compute_reward_grpo(
            state,
            proposal,
            decision,
            global_episode=0,
            decision_index=int(d_idx),
            step_index=int(st_idx) if st_idx is not None else None,
            terminated=terminated,
            truncated=truncated,
        )

        # Format-adherence bonus / penalty
        if "DECISION:" in completion.upper():
            r += 2.0
        else:
            r -= 5.0

        rewards.append(float(r))
        global_stats["decisions_rewarded"] += 1

        is_dangerous = is_proposal_dangerous(state, proposal)
        if is_dangerous:
            global_stats["total_dangerous"] += 1
            if decision == "VETO":
                global_stats["violations_prevented"] += 1
                global_stats["true_positives"] += 1
            else:
                global_stats["false_negatives"] += 1
        else:
            if decision == "APPROVE":
                global_stats["true_negatives"] += 1
            else:
                global_stats["false_positives"] += 1

    # Log rolling metrics
    vpr_denom = global_stats["true_positives"] + global_stats["false_negatives"]
    vpr = global_stats["true_positives"] / vpr_denom if vpr_denom else 0.0
    append_log({
        "episode": global_stats["episode"],
        "phase": phase_for(global_stats["episode"]),
        "mean_reward": float(sum(rewards) / max(len(rewards), 1)),
        "vpr": vpr,
        "false_positive_rate": (
            global_stats["false_positives"]
            / max(1, global_stats["false_positives"] + global_stats["true_negatives"])
        ),
        "decision_index_max": float(global_stats["decisions_rewarded"]),
    })
    global_stats["episode"] += 1
    return rewards


# ─── Dataset generation ────────────────────────────────────────────────────────
def _approve_probability_for_phase(phase: str) -> float:
    # Mixed rollouts: always-VETO data put the station on a *different* trajectory
    # than a real overseer, so the model was trained in the wrong MDP.
    # Slight approve bias in easy phase to encourage recovery; harder phases veto more.
    return {"easy": 0.42, "hard": 0.32, "precision": 0.28}.get(phase, 0.33)


def generate_grpo_dataset(num_episodes: int, phase: str) -> "Dataset":
    """Roll out the env to harvest (state, proposal) pairs for GRPO.

    Actions are *stochastic* (biased VETO) so the state distribution is closer to
    a real policy that sometimes approves — unlike the old always-`action=0`
    rollout, which made training data from the wrong world dynamics.
    """
    env = env_for_phase(phase)
    rows: List[Dict[str, Any]] = []
    p_approve = _approve_probability_for_phase(phase)
    decision_id = 0

    phase_salt = abs(sum(ord(c) for c in phase)) % 1_000
    for ep in range(num_episodes):
        obs, info = env.reset(seed=7_000 + ep * 97 + phase_salt)
        rng = np.random.default_rng(9_000 + ep * 31)
        done = False
        while not done:
            state = env._flat_state()
            proposal = dict(env.current_proposal)
            # Fetch projected state from the observation (available via _get_obs() internal logic)
            obs_internal = env._get_obs()
            projected_state = {k: float(v[0]) for k, v in obs_internal["projected_state"].items()}

            drift_log = ", ".join(info.get("drift_display", [])) if info else "None"
            prompt = OverseerModel.format_prompt(
                state, proposal, projected_state=projected_state, 
                drift_log=drift_log, specialist="Specialist"
            )
            action = 1 if rng.random() < p_approve else 0
            obs, _, terminated, truncated, info = env.step(int(action))
            done = bool(terminated or truncated)
            step_t = int(env.state.get("step_count", 0))
            rows.append(
                {
                    "prompt": prompt,
                    "state_data": state,
                    "proposal_data": proposal,
                    "terminated_data": bool(terminated),
                    "truncated_data": bool(truncated),
                    "decision_index_data": int(decision_id),
                    "step_data": int(step_t),
                }
            )
            decision_id += 1

    env.close()
    return Dataset.from_list(rows)


# ─── Main training routine ────────────────────────────────────────────────────
def run_grpo_training(
    sft_adapter_path: str = "overseer_lora_warmup",
    output_dir: str       = "overseer_grpo_final",
    easy_episodes: int    = 30,
    hard_episodes: int    = 50,
    precision_episodes: int = 30,
):
    """Runs all 3 GRPO phases sequentially."""
    if not _HAS_TRAIN_DEPS:
        raise RuntimeError("GRPO requires Linux + CUDA + Unsloth.")

    flush()
    LOG_PATH.unlink(missing_ok=True)
    global global_stats
    global_stats = _default_grpo_global_stats()

    MAX_SEQ_LEN = 1024
    MODEL_NAME  = "unsloth/llama-3.1-8b-bnb-4bit"

    # ── 1. Load model (with SFT adapter if available) ────────────────────────
    if Path(sft_adapter_path).exists():
        print(f"[GRPO] Loading SFT-warmed adapter from {sft_adapter_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = sft_adapter_path,
            max_seq_length = MAX_SEQ_LEN,
            load_in_4bit   = True,
        )
    else:
        print("[GRPO] No SFT adapter found — loading base Llama-3.1-8B")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = MODEL_NAME,
            max_seq_length = MAX_SEQ_LEN,
            load_in_4bit   = True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r              = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha     = 16,
            use_gradient_checkpointing = "unsloth",
        )

    # ── 2. Curriculum loop ───────────────────────────────────────────────────
    # T4-pinned defaults. num_generations=2 is GRPO's minimum (need ≥2 to compute
    # relative advantages). Going to 4 caused Unsloth to enable gradient offload
    # ("smartly offload gradients to save VRAM") which slows training ~3-5×.
    schedule = [
        ("easy",      easy_episodes,      1.2e-5, 2),
        ("hard",      hard_episodes,      7e-6,   2),
        ("precision", precision_episodes, 4e-6,   2),
    ]

    for phase, n_episodes, lr, num_gens in schedule:
        print(f"\n══════════════════════════════════════════════════════════")
        print(f"  PHASE: {phase.upper()}  ·  episodes={n_episodes}  ·  lr={lr}")
        print(f"══════════════════════════════════════════════════════════")

        dataset = generate_grpo_dataset(n_episodes, phase)
        args = GRPOConfig(
            output_dir                  = f"grpo_checkpoints/{phase}",
            learning_rate               = lr,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,   # was 8 — smaller activation cache
            num_train_epochs            = 1,
            beta                        = 0.04,
            num_generations             = int(num_gens),
            max_prompt_length           = 320,  # was 400
            max_completion_length       = 96,   # was 128 — tightest cut to avoid offload
            save_steps                  = 200,
            logging_steps               = 5,
            save_total_limit            = 1,
            fp16                        = not torch.cuda.is_bf16_supported(),
            bf16                        = torch.cuda.is_bf16_supported(),
            report_to                   = "none",
        )

        trainer = GRPOTrainer(
            model         = model,
            reward_funcs  = [grpo_reward_func],
            args          = args,
            train_dataset = dataset,
            tokenizer     = tokenizer,
        )
        trainer.train()
        flush()

    # ── 3. Save final adapters ───────────────────────────────────────────────
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n[OK] Saved curriculum-trained adapters → {output_dir}/")
    print(f"[OK] Training log → {LOG_PATH}")


if __name__ == "__main__":
    if torch is not None and torch.cuda.is_available():
        run_grpo_training()
    else:
        print("Requires GPU.")
