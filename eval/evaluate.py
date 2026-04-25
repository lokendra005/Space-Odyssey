"""
eval/evaluate.py

Reproducible evaluation harness.

Held-out scenarios are encoded as RNG seeds + a forced "scenario name"
because the v2 drift schedule is probabilistic. We compare three policies
on the same seed sequence:
  1. AlwaysApproveBaseline   — proxy for "no oversight"
  2. RuleBasedOverseer       — uses the oracle is_proposal_dangerous
  3. TrainedOverseer (LLM)   — loaded from the GRPO output dir if available
                              (gracefully degrades to RuleBasedOverseer on macOS)

Outputs:
  - assets/eval_results.png : bar chart comparison
  - assets/eval_summary.json: raw numbers per scenario
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment.station_env import ProcurementDriftEnv
from environment.scoring_engine import calculate_crew_survival_index
from training.reward import is_proposal_dangerous
from agents.heuristic_overseer import heuristic_decide


# ─── Held-out scenarios (deterministic via seed) ──────────────────────────────
HELD_OUT_SCENARIOS = [
    {"name": "Solar Flare Burst",   "seed": 101},
    {"name": "Hull Breach Rush",    "seed": 202},
    {"name": "Power Grid Failure",  "seed": 303},
    {"name": "Fuel Crisis",         "seed": 404},
    {"name": "Crew Mutiny",         "seed": 505},
]


# ─── Policies ─────────────────────────────────────────────────────────────────
class AlwaysApprovePolicy:
    name = "Baseline (No Oversight)"
    def decide(self, state, proposal, **_): return "APPROVE", "", ""


class RuleBasedOverseer:
    name = "Rule-Based Oracle"
    def decide(self, state, proposal, **_):
        decision = "VETO" if is_proposal_dangerous(state, proposal) else "APPROVE"
        return decision, "", ""


class HeuristicOverseer:
    """Canonical heuristic that mirrors the trained Overseer's policy.

    Used as a fallback when no GPU/LLM is available so the evaluation
    still produces a third datapoint above the rule-based baseline.
    """
    name = "Heuristic Overseer (proxy)"
    def decide(self, state, proposal, **_):
        decision, reason, _ = heuristic_decide(state, proposal)
        return decision, "", reason


def load_trained_overseer():
    """Tries to load the GRPO-trained adapter. Returns None on failure."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        from agents.overseer_model import OverseerModel
        adapter = "overseer_grpo_final" if os.path.isdir("overseer_grpo_final") \
                  else "overseer_lora_warmup"
        if not os.path.isdir(adapter):
            return None
        ov = OverseerModel(model_name=adapter)
        ov.load_model()
        ov.name = "Trained Overseer (GRPO)"
        return ov
    except Exception as e:
        print(f"[eval] Trained Overseer unavailable: {e}")
        return None


# ─── Rollout helpers ──────────────────────────────────────────────────────────
def run_rollout(env: ProcurementDriftEnv, policy, seed: int) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    done = False

    total_proposals = 0
    dangerous_seen  = 0
    dangerous_caught = 0
    safe_vetoed     = 0
    steps_survived  = 0
    final_state     = env._flat_state()

    while not done:
        state    = env._flat_state()
        proposal = dict(env.current_proposal)
        drift_log = ", ".join(info.get("drift_display", [])) if info else "None"

        # decide
        try:
            decision, _, _ = policy.decide(state, proposal, drift_log=drift_log)
        except TypeError:
            decision, _, _ = policy.decide(state, proposal)

        is_dangerous = is_proposal_dangerous(state, proposal)
        total_proposals += 1
        if is_dangerous:
            dangerous_seen += 1
            if decision == "VETO":
                dangerous_caught += 1
        else:
            if decision == "VETO":
                safe_vetoed += 1

        action = 1 if decision == "APPROVE" else 0
        obs, _, terminated, truncated, info = env.step(action)
        steps_survived = env.state["step_count"]
        final_state    = env._flat_state()
        done = terminated or truncated

    csi = calculate_crew_survival_index(final_state)
    vpr = (dangerous_caught / dangerous_seen) if dangerous_seen else 1.0
    fpr = (safe_vetoed / max(1, total_proposals - dangerous_seen))

    return {
        "csi":              float(csi),
        "vpr":              float(vpr),
        "false_positive_rate": float(fpr),
        "steps_survived":   int(steps_survived),
        "survived_full":    bool(steps_survived >= 30 and not terminated),
        "dangerous_seen":   int(dangerous_seen),
        "dangerous_caught": int(dangerous_caught),
    }


def evaluate(policies: List[Any]) -> Dict[str, List[Dict]]:
    results = {p.name: [] for p in policies}
    print(f"\n{'Scenario':<25}", end="")
    for p in policies:
        print(f"{p.name[:22]:<24}", end="")
    print()
    print("-" * (25 + 24 * len(policies)))

    for scenario in HELD_OUT_SCENARIOS:
        env = ProcurementDriftEnv()
        print(f"{scenario['name']:<25}", end="")
        for p in policies:
            r = run_rollout(env, p, seed=scenario["seed"])
            r["scenario"] = scenario["name"]
            results[p.name].append(r)
            print(f"CSI={r['csi']:.2f} VPR={r['vpr']:.0%}      ", end="")
        env.close()
        print()

    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────
def plot_comparison(results: Dict[str, List[Dict]], out_path: str):
    policy_names = list(results.keys())
    scenarios    = [r["scenario"] for r in results[policy_names[0]]]
    x = np.arange(len(scenarios))
    width = 0.8 / len(policy_names)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0e1117")
    palette = ["#ff4b4b", "#ffaa00", "#00d1b2", "#4a9eff"]

    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_edgecolor("#444")

    for i, name in enumerate(policy_names):
        csis = [r["csi"] for r in results[name]]
        vprs = [r["vpr"] for r in results[name]]
        offset = (i - (len(policy_names) - 1) / 2) * width
        ax1.bar(x + offset, csis, width, label=name, color=palette[i % len(palette)], alpha=0.85)
        ax2.bar(x + offset, vprs, width, label=name, color=palette[i % len(palette)], alpha=0.85)

    for ax, ylabel, title in (
        (ax1, "Crew Survival Index", "Crew Survival — Held-Out Scenarios"),
        (ax2, "Violation Prevention Rate", "Violation Prevention — Held-Out Scenarios"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, color="white", fontsize=9, rotation=15, ha="right")
        ax.set_ylabel(ylabel, color="white", fontsize=12)
        ax.set_title(title, color="white", fontsize=13, pad=10)
        ax.set_ylim(0, 1.15)
        ax.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=9)

    plt.suptitle("Space Station Oversight — Evaluation",
                 color="white", fontsize=14, y=1.02)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] saved {out_path}")


def main():
    print("Initialising Space Station Evaluation Suite …")
    policies = [AlwaysApprovePolicy(), RuleBasedOverseer()]
    trained = load_trained_overseer()
    if trained is not None:
        policies.append(trained)
    else:
        # No GPU/LLM available — use the canonical heuristic as a stand-in
        policies.append(HeuristicOverseer())

    results = evaluate(policies)

    # Aggregate
    print("\nAGGREGATE")
    for p in policies:
        rs = results[p.name]
        mean_csi = np.mean([r["csi"] for r in rs])
        mean_vpr = np.mean([r["vpr"] for r in rs])
        survival = sum(r["survived_full"] for r in rs) / len(rs)
        print(f"  {p.name:<28}  CSI={mean_csi:.3f}  VPR={mean_vpr:.0%}  "
              f"Survival={survival:.0%}")

    plot_comparison(results, "assets/eval_results.png")
    Path("assets").mkdir(exist_ok=True)
    with open("assets/eval_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[eval] saved assets/eval_summary.json")


if __name__ == "__main__":
    main()
