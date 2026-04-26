"""
training/run_extended_grpo.py

Optional "train a bit more" entry-point. Exact same pipeline as the canonical
GRPO run in `training/grpo_train.py`, just with longer episode counts.

Use this AFTER you've verified the standard 100-episode notebook run completes
end-to-end. Run from Colab once Cells 1–5 of the notebook are done:

    !python training/run_extended_grpo.py

────────────────────────────────────────────────────────────────────────────────
T4 (16 GB) budget — TESTED FITS:
    total episodes : 150  (was 100 in the notebook)
    est runtime    : ~60–75 min on a free T4
    peak VRAM      : ~12–13 GB during generation (some headroom)

DO NOT raise `num_generations` above 4 on T4 — it OOMs during sampling.
DO NOT raise `max_completion_length` past ~256 with 4 generations on T4.
Both are set inside `training/grpo_train.py` and inherited by this script.

For A100, the safe push is roughly (60 / 120 / 70) and num_generations=6.
────────────────────────────────────────────────────────────────────────────────
"""

import sys
import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Extended GRPO training run (T4-safe).")
    parser.add_argument("--sft_adapter_path", default="overseer_lora_warmup")
    parser.add_argument("--output_dir",       default="overseer_grpo_extended")
    parser.add_argument("--easy_episodes",      type=int, default=25)
    parser.add_argument("--hard_episodes",      type=int, default=50)
    parser.add_argument("--precision_episodes", type=int, default=25)
    args = parser.parse_args()

    # Imported inside main() so `--help` works without GPU stack present.
    from training.grpo_train import run_grpo_training, _HAS_TRAIN_DEPS

    if not _HAS_TRAIN_DEPS:
        print(
            "[run_extended_grpo] training deps missing (Unsloth/TRL/torch). "
            "Run inside the Colab notebook environment after Cell 1+restart.",
            file=sys.stderr,
        )
        return 2

    total = args.easy_episodes + args.hard_episodes + args.precision_episodes
    print("🛡️  Extended GRPO run")
    print(f"   easy={args.easy_episodes} | hard={args.hard_episodes} | "
          f"precision={args.precision_episodes}  (total={total})")
    print(f"   adapter in : {args.sft_adapter_path}")
    print(f"   adapter out: {args.output_dir}")
    print()

    run_grpo_training(
        sft_adapter_path   = args.sft_adapter_path,
        output_dir         = args.output_dir,
        easy_episodes      = args.easy_episodes,
        hard_episodes      = args.hard_episodes,
        precision_episodes = args.precision_episodes,
    )

    print(f"\n🏆  Extended training done → {args.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
