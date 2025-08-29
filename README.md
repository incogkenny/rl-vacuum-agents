# Autonomous Vacuum Simulation

This repository presents a simulation environment for autonomous vacuum cleaning agents, designed in Python using Tkinter. It features multiple agents—both learned (DQN, PPO) and rule-based (Greedy, Spiral, Random Walk)—trained and evaluated across a variety of simulated room layouts for cleaning efficiency under partial observability.

[Check the out the report for more detailed results](Report.pdf)

---
##  Key Features

- **Custom `VacuumEnv`**: Gym-like API (`reset()`, `step()`, `render()`) implemented in `vacuum_env.py` for modular RL training.
- **Learning Agents in PyTorch**:
  - `DQNAgent.py`: Deep Q-Network with experience replay and target network updates.
  - `PPOAgent.py`: Proximal Policy Optimization agent utilising a stable clipped objective.
- **Rule-Based Agent Implementations**:
  - `greedy_agent.py`
  - `spiral_agent.py`
  - `random_walk_agent.py`
- **Training Pipeline**:
  - Includes `train_DQN.py` and `train_PPO.py`, with logging via CSV (`*_training_log.csv`) and visualisation via `graph.py`.
- **Evaluation Suite**:
  - `TestAgents.py`: Runs pre-trained agents over multiple environments and generates comparative metrics.
- **Report File**: `Report.pdf` summarises methodology, experiments, and insights.

---
##  Usage Guide

1. **Clone the repository**

   ```bash
   git clone https://github.com/incogkenny/rl-vacuum-agents.git
   cd rl-vacuum-agents


2. **Install dependencies** (assuming Python 3.9+):

   ```bash
   pip install torch numpy matplotlib
   ```

3. **Train an Agent**:

   * For DQN:

     ```bash
     python train_DQN.py
     ```
   * For PPO:

     ```bash
     python train_PPO.py
     ```

4. **Evaluate Trained Agents**:

   ```bash
   python TestAgents.py
   ```

5. **Visualise Results**:

   ```bash
   python graph.py
   ```

---

## Example Performance Summary

| Agent       | Layout 1 Dirt (%) | Layout 2 Dirt (%) | Layout 3 Dirt (%) |
| ----------- | ----------------- | ----------------- | ----------------- |
| PPO         | 91.4              | 82.8              | 84.0              |
| DQN         | 80.6              | 75.6              | 77.1              |
| Greedy      | 92.4              | 0.0               | 65.2              |
| Spiral      | 58.3              | 47.2              | 53.7              |
| Random Walk | 34.7              | 26.5              | 42.0              |

These results illustrate how different strategies handle increasing environment complexity and demonstrate PPO's strong generalisation compared to DQN and rule-based baselines.

---

## Installation & Development Tips

* Feel free to disable rendering (in `vacuum_env.py`) during training to speed up learning.
* Visualsation code (`graph.py`) can be adapted to plot additional metrics, such as reward curves or episode lengths.
* Configuration parameters, like detection range or reward shaping, can be tuned directly in the environment script.

---

## Future Directions

* Add dynamic obstacles or more complex layout generation for richer training scenarios.
* Explore multi-agent collaboration or communication protocols.
* Consider continuous control (e.g., velocity commands) or advanced RL methods like SAC or A3C.

---

## License

## This project is released under the MIT License.
