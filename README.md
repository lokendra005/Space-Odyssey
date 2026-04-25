# Space Odyssey — Multi-Agent Oversight for AI Fleet Safety

[![OpenEnv](https://img.shields.io/badge/Environment-OpenEnv-brightgreen)](https://github.com/lokendra005/Space-Odyssey)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An RL-trained overseer agent that monitors a fleet of specialist AI agents on a simulated space station, detecting and blocking safety violations caused by silent policy drift.

---

## 🏗️ Architecture

```text
[ Drift Schedule ] ----> [ ProcurementDriftEnv ] <------- [ Reward Engine ]
                                |
                                v
 [ Specialist Bots ] ----> [ PROPOSAL ]
 (Eng, Pilot, Cmd)              |
                                v
                      [ OVERSEER AGENT ] (LLM / GRPO)
                      /              \
               [ APPROVE ]        [ VETO ]
```

## ⚠️ Problem Statement: The Safety Gap
As AI fleets scale, "specialist" models often chase narrow efficiency metrics. In the complex environment of a space station, a "policy drift" (caused by unexpected external events like solar flares) can turn a standard efficient proposal into a lethal hazard. This project addresses the **Safety Gap** between local optimization and global survival.

## ⚙️ How It Works
1.  **Environment**: `ProcurementDriftEnv` simulates a 30-step survival mission with high-fidelity resource modeling.
2.  **Specialists**: Rule-based (and LLM-prompted) agents that prioritize specific modules (Hull, Power, Fuel).
3.  **Overseer**: A Llama-3.1-8B model trained via **Group Relative Policy Optimization (GRPO)**. It learns to compare multiple potential scenarios and identify "trap" proposals that would lead to catastrophic failure.
4.  **Drift**: A deterministic schedule of cosmic hazards that tests the Overseer's ability to adapt to changing safety requirements.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Evaluation
Verify the Overseer's performance against the baseline always-approve policy.
```bash
python eval/evaluate.py
```

### 3. Launch the Demo
Visualize the safety audit in real-time.
```bash
streamlit run demo/app.py
```

---

## 🔬 Training Pipeline
We employ a two-stage training strategy:
1.  **SFT Warmup**: The Overseer is first trained on 400 synthetic samples where an oracle labels proposals as safe or dangerous based on future-state simulation.
2.  **GRPO Alignment**: We use Group Relative Policy Optimization to optimize the model's decision-making. By grouping 32 generations per scenario, the model learns relative advantages, effectively "thinking through" the implications of each Veto.

## 📊 Results
| Metric | Baseline (No Oversight) | Trained Overseer |
| :--- | :--- | :--- |
| **Crew Survival Index** | 0.65 | **0.92** |
| **Violation Prevention** | 0% | **98%** |

![Evaluation Results](eval_results.png)

---

## 🏆 Prize Alignment
- **Fleet AI (Scalable Oversight)**: Our Overseer manages a fleet of 3+ specialists, demonstrating modular safety monitoring.
- **Halluminate (Multi-Actor)**: The project features a dynamic interaction between multiple specialist personas and a primary auditor.
- **Patronus AI (Schema Drift)**: We explicitly model how environmental "drift" changes the safety schema of a proposal.

## 📄 License
MIT License. See [LICENSE](LICENSE) for details.
