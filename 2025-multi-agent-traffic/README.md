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

**📦 Project Structure**
<img width="504" height="163" alt="image" src="https://github.com/user-attachments/assets/3ddebc57-5a4b-4b91-8b8a-274be2ab0d8b" />

**🧠 Methodology Overview**

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

**🛠️ Installation**
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

**▶️ Training**
Run MARL training with PPO:

bash
python training/train_marl_agents.py --config configs/ppo_marl.yaml
This will:

Initialize MetaDrive multi-agent environment

Launch Ray rollout workers

Train AGENT1 & AGENT2 adversarial policies

Save checkpoints in policies/

**🎯 Evaluation**
Evaluate trained policies:

bash
python evaluation/eval_policies.py --checkpoint policies/best_policy/
You can enable rendering:

bash
--render True
**📊 Results Summary**
Trained MARL agents successfully learned to:

##Perform rear‑end collisions
Before training, the traffic agents exhibited mostly random or non‑targeted behavior and rarely produced consistent rear‑end collisions with the ego vehicle.

![demo](https://github.com/user-attachments/assets/1a7552b1-4c78-40f8-9a3f-e6d7bc54d61a)

After training, the MARL agents learned to intentionally perform rear‑end collisions by closing the gap aggressively, maintaining high relative speed, and exploiting the ego vehicle’s conservative behavior.
![scenario_0](https://github.com/user-attachments/assets/f2805e94-2e17-4cd0-9054-2ec6dbee99fa)

**Execute cut‑ins and cut‑offs**
Before training, the traffic agents behaved randomly and were unable to perform structured lateral maneuvers. Lane changes occurred sporadically, without awareness of the ego vehicle’s position or timing, resulting in unrealistic or non‑adversarial interactions.
![demo](https://github.com/user-attachments/assets/b2335013-72ff-4f23-a86a-536c24962c46)

After training, the MARL agents learned to execute deliberate and well‑timed cut‑ins and cut‑offs:
Cut‑ins: Agents merge sharply into the ego vehicle’s lane with minimal headway, forcing the ego vehicle to brake or adjust its trajectory.
Cut‑offs: Agents accelerate, overtake, and then re‑enter the lane directly in front of the ego vehicle, reducing time‑to‑collision and creating a high‑pressure scenario.

![scenario_0](https://github.com/user-attachments/assets/f96a0a76-83e6-46cc-9b7d-995ade6ee4b5)

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
