**🚀 Key Features**

Multi‑Agent Reinforcement Learning (MARL) setup with PPO

Adaptive adversarial traffic agents capable of:

Cut‑in / cut‑off maneuvers

Tailgating & forced merges

Rear‑end collisions

Aggressive overtaking

IDM‑controlled ego vehicle for baseline AV behavior

Procedurally generated road networks (straight roads, intersections, roundabouts, etc.)

Custom reward & termination functions for adversarial behavior shaping

Distributed training using Ray RLlib (CPU + GPU support)

Policy checkpointing & evaluation in both trained and unseen scenarios

Generalization tests across different map structures

📦 Project Structure
Code
├── configs/                # Environment & PPO configuration files
├── env/                    # Custom MetaDrive multi-agent environment
├── policies/               # Saved PPO checkpoints
├── training/               # Training scripts (Ray RLlib)
├── evaluation/             # Scenario evaluation scripts
├── utils/                  # Helper functions (logging, plotting, etc.)
└── README.md               # Project documentation
🧠 Methodology Overview
1. Environment Setup
Built on MetaDrive with procedural map generation

Ego vehicle uses Intelligent Driver Model (IDM)

Two nearest traffic vehicles are selected as MARL agents

Agents receive LiDAR‑like observations (72–240 dims)

2. Reward Design
Agents are rewarded for:

Reducing distance to the ego vehicle

Performing cut‑ins, cut‑offs, overtakes

Maintaining forward progress

Penalties for:

Collisions with non‑ego vehicles

Leaving the drivable area

Crashing into static objects

3. Training
PPO with clipped objective

Distributed rollout workers

50+ iterations × 10 sessions

Best policy selected via reward convergence

4. Evaluation
Replaying trained policies in:

Straight roads

Roundabouts

Novel procedural maps

Measuring adversarial behavior consistency

Visualizing trajectories & interactions

🛠️ Installation
1. Clone the repository
bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
2. Create environment
bash
conda create -n marl-traffic python=3.9
conda activate marl-traffic
3. Install dependencies
bash
pip install -r requirements.txt
Dependencies include:

MetaDrive

Ray + RLlib

PyTorch

NumPy / Pandas

Matplotlib / Seaborn

▶️ Training
Run MARL training with PPO:

bash
python training/train_marl_agents.py --config configs/ppo_marl.yaml
This will:

Initialize MetaDrive multi-agent environment

Launch Ray rollout workers

Train AGENT1 & AGENT2 adversarial policies

Save checkpoints in policies/

🎯 Evaluation
Evaluate trained policies:

bash
python evaluation/eval_policies.py --checkpoint policies/best_policy/
You can enable rendering:

bash
--render True
📊 Results Summary
Trained MARL agents successfully learned to:

Perform rear‑end collisions

Execute cut‑ins and cut‑offs

Coordinate multi‑agent maneuvers

Generalize to unseen roundabout scenarios

The ego vehicle (IDM) exhibited:

Hesitation under adversarial pressure

Limited ability to avoid rear‑end threats

Reduced maneuverability in multi‑agent traps

These results demonstrate the effectiveness of MARL for generating realistic, safety‑critical scenarios.

📘 Citing This Work
If you use this repository in academic work, please cite:

Code
Joy, Maria Shaima. 
"Development of learning-based multi-agent models for validating automated driving functions in adaptive traffic simulations."
Master’s Thesis, Karlsruhe University of Applied Sciences, 2025.
🤝 Contributing
Contributions are welcome!
Please open an issue or submit a pull request.

📄 License
Specify your license here (MIT, Apache 2.0, etc.)
