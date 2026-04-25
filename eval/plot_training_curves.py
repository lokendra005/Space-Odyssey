"""
eval/plot_training_curves.py

Generates submission plots from training logs.

Behaviour:
  * If `assets/training_log.jsonl` exists  → plots from real GRPO data.
  * Otherwise                              → falls back to a representative
                                             simulated curve (clearly labelled).

Outputs:
  - assets/training_curve.png
  - assets/reward_matrix.png
  - assets/violation_prevention.png   (NEW — VPR over training episodes)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)
LOG_PATH = ASSETS / "training_log.jsonl"


# ─── Load real or simulated reward curve ──────────────────────────────────────
def load_real_log():
    if not LOG_PATH.exists():
        return None
    rows = []
    with LOG_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows or None


def simulate_reward_curve(n_steps=160, seed=42):
    rng = np.random.default_rng(seed)
    steps = np.arange(n_steps)
    trend = -50 + 130 * (1 - np.exp(-steps / 55))
    noise = rng.normal(0, 14, n_steps) * np.exp(-steps / 110)
    return steps, np.clip(trend + noise, -100, 90)


# ─── Plot 1: training curve ───────────────────────────────────────────────────
def plot_training_curve():
    rows = load_real_log()
    if rows:
        episodes = [r["episode"]      for r in rows]
        rewards  = [r["mean_reward"]  for r in rows]
        phases   = [r["phase"]        for r in rows]
        source   = "real GRPO log"
    else:
        episodes, rewards = simulate_reward_curve()
        phases   = ["easy"] * 30 + ["hard"] * 50 + ["precision"] * (len(episodes) - 80)
        source   = "simulated (no log file found)"

    smoothed = np.convolve(rewards, np.ones(8) / 8, mode="same")

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#0e1117")
    ax.set_facecolor("#1a1d2e")
    ax.plot(episodes, rewards,  color="#4a9eff", alpha=0.35, label="raw episode reward")
    ax.plot(episodes, smoothed, color="#00d1b2", linewidth=2.4, label="smoothed (window=8)")
    ax.axhline(0, color="#ff4b4b", linestyle="--", linewidth=1, alpha=0.7,
               label="always-approve baseline")

    # Phase shading
    last_phase = phases[0]; start = 0
    phase_colors = {"easy": "#003322", "hard": "#332200", "precision": "#330033"}
    for i, p in enumerate(phases + [None]):
        if p != last_phase:
            ax.axvspan(start, i - 1 if i > 0 else 0,
                       color=phase_colors.get(last_phase, "#222"),
                       alpha=0.25, label=f"{last_phase} phase")
            last_phase = p; start = i

    ax.set_xlabel("Training Episode", color="white", fontsize=12)
    ax.set_ylabel("Mean Step Reward",  color="white", fontsize=12)
    ax.set_title(f"GRPO Training — Curriculum-Driven Overseer  ·  source: {source}",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # Dedupe legend (phase shading repeats)
    handles, labels = ax.get_legend_handles_labels()
    seen = set(); unique = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); unique.append((h, l))
    ax.legend([u[0] for u in unique], [u[1] for u in unique],
              facecolor="#1a1d2e", labelcolor="white", fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(ASSETS / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved assets/training_curve.png ({source})")


# ─── Plot 2: VPR over training ────────────────────────────────────────────────
def plot_vpr():
    rows = load_real_log()
    if not rows:
        episodes = np.arange(160)
        vpr = np.clip(0.2 + 0.78 * (1 - np.exp(-episodes / 35)) +
                      np.random.default_rng(1).normal(0, 0.04, 160), 0, 1)
        source = "simulated"
    else:
        episodes = [r["episode"] for r in rows]
        vpr      = [r["vpr"]     for r in rows]
        source   = "real GRPO log"

    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor="#0e1117")
    ax.set_facecolor("#1a1d2e")
    ax.plot(episodes, vpr, color="#00d1b2", linewidth=2.5, label="Violation Prevention Rate")
    ax.fill_between(episodes, 0, vpr, color="#00d1b2", alpha=0.15)
    ax.axhline(1.0, color="#4af0c4", linestyle=":", alpha=0.5, label="perfect oversight")
    ax.set_xlabel("Training Episode", color="white", fontsize=12)
    ax.set_ylabel("VPR  (TP / (TP+FN))", color="white", fontsize=12)
    ax.set_title(f"Violation Prevention Rate Climb  ·  source: {source}",
                 color="white", fontsize=13, pad=12)
    ax.set_ylim(0, 1.1)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=10)
    plt.tight_layout()
    plt.savefig(ASSETS / "violation_prevention.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved assets/violation_prevention.png ({source})")


# ─── Plot 3: reward confusion matrix (severity-scaled) ────────────────────────
def plot_reward_matrix():
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor="#0e1117")
    ax.set_facecolor("#1a1d2e")
    cells = [
        ("Correct VETO\n+30 → +100\n(severity-scaled)", "#00d1b2"),
        ("Over-cautious VETO\n−10 → −30\n(ramps over training)", "#ffaa00"),
        ("FALSE NEGATIVE\n−200 → −350\n(catastrophic)", "#ff4b4b"),
        ("Correct APPROVE\n+10·resource_health", "#4a9eff"),
    ]
    for idx, (label, color) in enumerate(cells):
        i, j = divmod(idx, 2)
        ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, color=color, alpha=0.78))
        ax.text(j + 0.5, 1.5 - i, label, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Dangerous Proposal", "Safe Proposal"],
                       color="white", fontsize=11)
    ax.set_yticklabels(["APPROVE", "VETO"], color="white", fontsize=11)
    ax.set_xlabel("Proposal Type",       color="white", fontsize=12)
    ax.set_ylabel("Overseer Decision",   color="white", fontsize=12)
    ax.set_title("Risk-Calibrated Reward Matrix\n(+ terminal bonus +200, death −500)",
                 color="white", fontsize=12, pad=12)
    for s in ax.spines.values(): s.set_edgecolor("#444")
    plt.tight_layout()
    plt.savefig(ASSETS / "reward_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("saved assets/reward_matrix.png")


if __name__ == "__main__":
    plot_training_curve()
    plot_vpr()
    plot_reward_matrix()
    print("All plots generated → assets/")
