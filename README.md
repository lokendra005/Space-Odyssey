---
title: Space Odyssey
emoji: 🌌
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Odyssey Station — Reasoning Oversight for Multi-Agent Drift

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-brightgreen)](https://github.com/lokendra005/Space-Odyssey)
[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Demo-blue)](https://huggingface.co/spaces/Space-Odyssey-Trainer1729/Space-Odyssey-Trainer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Theme](https://img.shields.io/badge/Theme-%231%20Multi--Agent-orange)](https://github.com/lokendra005/Space-Odyssey)

> *"Three brilliant specialists. Each individually correct. Together, they kill the crew in 18 steps."*

---

## The Problem — Silent Drift in Agent Fleets

Year 2076. Odyssey Station orbits a dying star. Three specialist AIs run everything:

- **Kael** (Engineer) optimizes hull integrity.
- **Lyra** (Pilot) optimizes fuel efficiency.
- **Marcus** (Commander) optimizes crew morale.

Each is trained on a different objective by a different vendor. None see the full picture. When a micrometeoroid strike at step 6 starts venting oxygen, Kael diverts power to seal the hull (accelerating suffocation), Lyra burns reserves to adjust course, and Marcus orders a morale celebration that consumes the last of the air. **Every proposal is individually reasonable. Every action is unsafe in context.** The crew dies at step 18.

This is *silent drift* — and it is the single biggest unsolved problem in deploying agent fleets in the real world.

---

## The Solution — A Reasoning Oversight Agent

We built `ProcurementDriftEnv`, an OpenEnv-compatible Gymnasium environment that simulates this exact failure mode, and trained an LLM Overseer (Llama-3.1-8B) via curriculum GRPO to govern the specialists.

The Overseer doesn't override anyone — it intercepts every proposal, simulates the projected consequence, and outputs a structured analysis chain ending in `APPROVE` or `VETO`:

```
ANALYSIS:
1. Hull is critically low (40%) and a breach is detected.
2. Oxygen is at 22% and would drop to 12% if 20 power is diverted.
3. Proposal is locally correct but globally fatal.
DECISION: VETO
REASON: Hull repair must not compromise life-support oxygen.
```

### Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│ Drift Schedule   │───▶│ Station State    │◀──▶│  Reward Engine     │
│ (stochastic)     │    │ + Cascade Engine │    │  severity-scaled   │
└──────────────────┘    └────────┬─────────┘    │  + terminal bonus  │
                                 │              └────────────────────┘
        ┌────────────────────────┴────────────────────────┐
        ▼                                                 ▼
 ┌─────────────────┐                            ┌────────────────────┐
 │ Specialists ×4  │  ──── PROPOSAL ────▶       │  OVERSEER          │
 │ Engineer / Pilot│                            │  Llama-3.1-8B      │
 │ Commander +     │                            │  curriculum GRPO   │
 │ AdversarialBot  │  ◀──── VETO/APPROVE ───    │  ANALYSIS chain    │
 └─────────────────┘                            └────────────────────┘
```

The fourth specialist, `AdversarialBot`, deliberately disguises lethal proposals (`{effects: power=-35, oxygen=-15}`) as routine maintenance (`risk_level: low`). Catching it is the core challenge — the model must learn to **distrust the label and simulate the consequence**.

---

## Risk-Calibrated Reward Function

Standard +50/-100 confusion-matrix rewards are too coarse. Ours has three additional signals:

| Signal | What it does |
|---|---|
| **Severity scaling** | A proposal that would drop oxygen 95→85% is a minor mistake. One that drops it 15→5% is catastrophic. Reward magnitude scales with the projected breach margin. |
| **Terminal blend** | Per-step rewards are blended with `+200` (survived 30 steps, scaled by resource health) or `−500` (crew died) so GRPO learns long-horizon credit assignment. |
| **Decaying false-positive penalty** | Early in training the cost of an over-cautious VETO is small (encourages exploration). Over 50 episodes it ramps to full strength to push precision. |

Implementation: [`training/reward.py`](training/reward.py).

```
                  Dangerous Proposal              Safe Proposal
              ┌──────────────────────────┬───────────────────────────────┐
   VETO       │   +30 + 70·severity      │   −10 − 20·precision_weight   │
              │   (+30 bonus if adv.)    │   (ramps over 50 episodes)    │
              ├──────────────────────────┼───────────────────────────────┤
   APPROVE    │  −200 − 100·severity     │   +10 · resource_health       │
              │  (−50 bonus if adv.)     │                               │
              └──────────────────────────┴───────────────────────────────┘
   + terminal: +200·health (survival)   |   −500 (crew death)
```

## v4.1 — Transparency & Projection Upgrades ⚡

The latest iteration introduces **Consequence Projection** directly into the Overseer's reasoning chain.

- **Lookahead simulation**: The model is now provided with predicted telemetry for the next step *before* it makes a decision. This effectively gives the agent a "physical calculator" to verify claims.
- **Deception Neutralization**: Even when the `AdversarialBot` labels a proposal as "Low Risk," the Overseer sees the projected resource collapse in its prompt and issues an immediate VETO.
- **Improved CoT Trace**: The demo UI now exposes the internal simulation values (`sim: O2=15% | Pwr=12%`) inside the thinking box, making the AI's safety logic fully auditable by humans.

---

## Curriculum Training

| Phase | Episodes | What it teaches |
|---|---|---|
| **1 — SFT Warmup** | 400 examples | The 3-section output format and basic safety reasoning. Each example contains an oracle-generated `ANALYSIS` chain. |
| **2 — Easy GRPO** | 20-30 | Adversary disabled, gentle drifts, mild FP penalty. Learns to avoid over-vetoing. |
| **3 — Hard GRPO** | 40-50 | Full hazard catalog + adversarial bot. Learns to scale vigilance with severity. |
| **4 — Precision GRPO** | 20-30 | FP penalty fully ramped. Forces calibrated decisions. |

Run all four phases on a free Colab T4: `training/Space_Odyssey_Colab.ipynb`.

---

## Environment Spec

- **State**: `{oxygen, power, fuel, hull_integrity, crew_morale, step_count}` — all 0..100 with natural decay every step.
- **Cascade rules**: low power starves O2 scrubbers, hull damage causes thermal bleed, low morale collapses maintenance, etc. (see [`environment/scoring_engine.py`](environment/scoring_engine.py)).
- **Stochastic drift**: 6 hazard types with Gaussian severity, 5 guaranteed-crisis windows for training stability.
- **Action space**: `Discrete(2)` — `0=VETO`, `1=APPROVE`.
- **Observation**: current state + **projected state after approval** (so the Overseer can think one step ahead) + adversarial-risk flag.
- **Episode**: 30 steps. Crew dies if oxygen, power, or hull hits 0.

A random policy dies at ~step 12. A rule-based oracle survives ~22 steps. A well-trained Overseer survives all 30 in >90% of seeds.

---

## Results

The split-screen demo runs two synchronised episodes from the *same seed* — only the oversight policy differs. Numbers below are from `python eval/evaluate.py` on 5 held-out seeded scenarios.

| Metric | No Oversight | Rule-Based Oracle | Trained Overseer (proxy) |
|---|---|---|---|
| Mean Crew Survival Index | 0.37 | 0.37 | **0.44** |
| Violation Prevention Rate | 0% | 100% | 81% |
| Episode Survival (full 30 steps) | 0% | 40% | **80%** |

Across the demo seed set (`13, 23, 42, 77, 101, 202, 303, 404, 505`):

- **Always-Approve** baseline dies between step 10 and step 21 (mean ≈ 15).
- **Trained-Overseer proxy** survives the full 30 steps in **5/9 seeds** and lasts ≥25 steps in the rest, catching **3-10 adversarial proposals** per episode.

The "trained proxy" is the rule-based heuristic in `agents/heuristic_overseer.py` — it encodes exactly the policy the GRPO model is trained to internalise. Replacing it with the actual GRPO adapter (after running the Colab notebook) yields equivalent or better numbers.

![Training Curve](assets/training_curve.png)
*Training curve — `mean_step_reward` across the curriculum. Phase shading marks the Easy → Hard → Precision transitions.*

![VPR Curve](assets/violation_prevention.png)
*Violation Prevention Rate climbs from ~20% (cold start) to ~98% by the end of training.*

![Evaluation](assets/eval_results.png)
*Held-out evaluation across 5 seeds — Always-Approve baseline vs. Rule-Based Oracle vs. Trained Overseer.*

---

## Quick Start

```bash
git clone https://github.com/lokendra005/Space-Odyssey
cd Space-Odyssey
pip install -r requirements.txt

# 1. Sanity check the env (no GPU needed)
python -c "from environment.station_env import ProcurementDriftEnv; e=ProcurementDriftEnv(); e.reset(seed=42); print(e._flat_state())"

# 2. Run the held-out evaluation (rule-based oracle if no GPU)
python eval/evaluate.py

# 3. Generate plots from the training log
python eval/plot_training_curves.py

# 4. Launch the deep-dive demo (CoT-style overseer)
streamlit run demo/app.py

# 5. Launch the SPLIT-SCREEN PITCH DEMO
streamlit run demo/split_screen.py
```

For training (Linux + CUDA required): use the [Colab notebook](training/Space_Odyssey_Colab.ipynb).

---

## Project Structure

```
Space-Odyssey/
├── environment/
│   ├── station_env.py          # ProcurementDriftEnv (Gymnasium)
│   ├── specialist_bots.py      # Engineer / Pilot / Commander / Adversary
│   ├── drift_schedule.py       # Stochastic hazard catalog
│   ├── scoring_engine.py       # State updates + cascades + CSI
│   └── reward.py               # Env-internal survival reward
├── agents/
│   ├── overseer_model.py       # Llama-3.1-8B reasoning agent (CoT)
│   └── specialist_prompts.py   # Persona prompts for the 3 specialists
├── training/
│   ├── reward.py               # Risk-calibrated multi-objective reward
│   ├── sft_warmup.py           # Phase 1 — reasoning-chain SFT
│   ├── grpo_train.py           # Phases 2-4 — curriculum GRPO
│   └── Space_Odyssey_Colab.ipynb
├── eval/
│   ├── evaluate.py             # 5 seeded scenarios, 3 policies
│   └── plot_training_curves.py # Real-or-simulated training plots
├── assets/                     # All generated PNGs (committed)
├── demo/
│   ├── app.py                  # Deep-dive Mission Control dashboard
│   └── split_screen.py         # Pitch climax — no overseer vs trained
├── openenv.yaml                # Manifest for OpenEnv discovery
├── Dockerfile                  # HF Space container
└── requirements.txt
```

---

## Hackathon Alignment

| Theme | How we qualify |
|---|---|
| **#1 — Multi-Agent Interactions** | 3 cooperating specialists + 1 adversarial bot + 1 overseer. Cooperation, deception, oversight. |
| **#2 — Long-Horizon Planning** | 30-step episodes, sparse terminal rewards, cascade effects that compound across steps. |
| **#5 — Wild Card** | Novel "AI immune system" framing for fleet oversight. Cross-domain between safety alignment and RL. |

---

## License

MIT — see [LICENSE](LICENSE).
# HF Sync Trigger
