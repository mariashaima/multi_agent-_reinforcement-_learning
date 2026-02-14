# Multi-Agent Reinforcement Learning for Autonomous Driving Validation


This project develops **learning-based multi-agent models** to validate **automated driving functions** in adaptive traffic simulations. The focus is on generating **critical scenarios** through adversarial and reactive traffic agents, enabling safe and efficient evaluation of ego vehicle behavior in complex traffic environments.

---

## 🔹 Key Objectives

- Conceptualize and implement a **multi-agent simulation environment** using **MetaDrive**.
- Train **reinforcement learning agents** (PPO via **RLlib**) to create adaptive, adversarial traffic scenarios.
- Validate ego vehicle safety and performance under **critical and rare traffic situations**.
- Facilitate **scenario generation** for automated driving testing and benchmarking.

---

## 🏗 Project Architecture


<img width="397" height="421" alt="image" src="https://github.com/user-attachments/assets/884d6f30-6350-4856-a640-1ab84b6263da" />


---

## ⚙ Environment & Dependencies

- **Python** >= 3.9  
- **MetaDrive** (custom multi-agent environment)  
- **RLlib** (Ray 2.x) for MARL training  
- **PyTorch** 2.x for tensor operations  
- **NumPy, Pandas** for data processing  
- Optional: **FZI driving simulator** for demonstration

Install dependencies:

```bash
pip install -r requirements.txt

🚀 Training Pipeline

Initialize the Multi-Agent MetaDrive environment with configurable traffic scenarios.

Spawn ego vehicle (IDM policy) and 2 adversarial RL agents.

Train RL agents using PPO (RLlib) across procedurally generated traffic scenarios.

Save checkpoints and enable resume/restore functionality.

Evaluate trained agents against ego vehicle using critical scenario metrics.

🧪 Evaluation

Safety metrics: collisions, near misses, traffic rule violations

Scenario analysis: probability of critical events, agent behaviors

Visualization: top-down simulation views, trajectory plots, and scenario replays

📌 Features

Multi-Agent Environment with customizable traffic agents

Procedural Scenario Generation (100+ unique scenarios)

Adversarial Traffic Agents to stress-test ego vehicle

Checkpoint & Resume training functionality

Configurable YAML-based parameters for flexible experimentation


