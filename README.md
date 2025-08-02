# Hospital Navigation RL Environment

A reinforcement learning environment for training AI agents to navigate hospital corridors, collect patients, and deliver them to appropriate medical departments while managing medication delivery in a telemedicine platform context.

## 🏥 Environment Overview

<img width="1007" height="709" alt="Image" src="https://github.com/user-attachments/assets/ace66f6d-4c09-4aea-8c96-b93ff8dabe5e" />


This environment simulates a realistic hospital layout where an AI agent must:

- **Navigate through hospital corridors** (cannot enter rooms directly)
- **Collect patients** from corridor locations with different urgency levels
- **Obtain medications** from pharmacy drug stations when patients need them
- **Deliver patients** to appropriate medical departments
- **Maximize patient care efficiency** while minimizing time and resources

### Key Features

- **Realistic Hospital Layout**: Multiple departments including Emergency, ICU, Surgery, Cardiology, Neurology, Pediatrics, Lab, Radiology, and Pharmacy
- **Corridor-Only Navigation**: Agent must stick to realistic pathways between rooms
- **Multi-Objective Task**: Balance between patient collection, drug delivery, and room assignment
- **Dynamic Patient Spawning**: Patients appear with varying urgency and medication needs
- **Visual Feedback**: Pygame-based rendering with detailed hospital visualization

## 🎯 Mission Context

This environment supports the development of AI systems for telemedicine platforms and digital AI labs by:

1. **Route Optimization**: Training agents to find efficient paths in complex healthcare facilities
2. **Resource Management**: Learning to prioritize tasks based on patient urgency and medication needs
3. **Decision Making**: Balancing multiple objectives in time-critical healthcare scenarios
4. **Workflow Automation**: Optimizing patient flow and medication delivery processes

## 📁 Project Structure

```
hospital-navigation-rl/
├── hospital_env.py          # Main environment implementation
├── dqn_agent.py            # Deep Q-Network agent
├── ppo_agent.py            # Proximal Policy Optimization agent
├── reinforce_agent.py      # REINFORCE algorithm agent
├── train.py                # Training script for all algorithms
├── play.py                 # Random agent demonstration
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── models/                # Saved model directory
├── results/              # Training results and plots
└── demos/               # Generated demonstration GIFs
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hospital-navigation-rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Demonstrations

View the environment with a random agent:
```bash
python play.py --mode random --episodes 3
```

View strategic random agent:
```bash
python play.py --mode strategic --episodes 3
```

Show static hospital layout:
```bash
python play.py --mode layout
```

### Training Agents

Train all three algorithms and compare:
```bash
python train.py --algorithm all --episodes 1000
```

Train specific algorithm:
```bash
python train.py --algorithm dqn --episodes 1000
python train.py --algorithm ppo --episodes 1000
python train.py --algorithm reinforce --episodes 1000
```

Evaluate trained model:
```bash
python train.py --algorithm dqn --evaluate --render
```

## 🤖 Reinforcement Learning Algorithms

### 1. Deep Q-Network (DQN) - Value-Based Method

**Characteristics:**
- **Type**: Off-policy, value-based
- **Action Selection**: Epsilon-greedy with experience replay
- **Network**: Deep neural network approximating Q-values
- **Training**: Uses target network and experience replay buffer

**Advantages:**
- Sample efficient through experience replay
- Stable learning with target network
- Works well in discrete action spaces

**Implementation Features:**
- Experience replay buffer (10,000 transitions)
- Target network updated every 100 steps
- Epsilon decay from 1.0 to 0.01
- Gradient clipping for stability

### 2. Proximal Policy Optimization (PPO) - Policy-Based Method

**Characteristics:**
- **Type**: On-policy, actor-critic
- **Action Selection**: Stochastic policy with probability sampling
- **Network**: Shared network with actor and critic heads
- **Training**: Uses clipped surrogate objective

**Advantages:**
- More stable than vanilla policy gradient
- Good sample efficiency
- Handles continuous and discrete actions well

**Implementation Features:**
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective (ε = 0.2)
- Multiple epochs per update (4 epochs)
- Entropy regularization for exploration

### 3. REINFORCE - Policy Gradient Method

**Characteristics:**
- **Type**: On-policy, policy gradient
- **Action Selection**: Direct policy optimization
- **Network**: Policy network outputting action probabilities
- **Training**: Monte Carlo policy gradient

**Advantages:**
- Simple and intuitive
- Direct policy optimization
- Works with stochastic policies

**Implementation Features:**
- Monte Carlo returns calculation
- Baseline subtraction (return normalization)
- Gradient clipping for stability
- Episode-based updates

## 🎮 Environment Details

### State Space (Observation)
The observation vector contains:
- **Agent Position**: Normalized x, y coordinates (2 values)
- **Agent Status**: Carrying patient flag, has drugs flag (2 values)
- **Patient Information**: Up to 6 patients with position, urgency, drug needs (24 values)
- **Drug Station Status**: Availability of 2 drug stations (2 values)

**Total Observation Size**: 30 dimensions

### Action Space
8 discrete actions representing movement directions:
- 0: Up, 1: Down, 2: Left, 3: Right
- 4: Up-Left, 5: Up-Right, 6: Down-Left, 7: Down-Right

### Reward Structure

**Positive Rewards:**
- +10: Collecting drugs from pharmacy
- +40: Delivering drugs to patient needing medication
- +15: Picking up patient
- +30-75: Delivering patient to correct department (based on urgency)

**Negative Rewards:**
- -0.1: Each step (efficiency incentive)
- -1: Staying in same position
- -30: Delivering patient to wrong department

### Episode Termination
Episodes end when:
- Maximum steps reached (1000 steps)
- All patients have been saved
- User closes the window (in demo mode)

## 📊 Performance Metrics

The training system tracks:
- **Episode Rewards**: Total reward accumulated per episode
- **Patients Saved**: Number of patients successfully delivered
- **Training Loss**: Algorithm-specific loss functions
- **Efficiency**: Patients saved per 1000 steps

## 🔧 Customization

### Environment Parameters
Modify `hospital_env.py` to adjust:
- Hospital layout and room positions
- Number and placement of drug stations
- Patient spawn rates and characteristics
- Reward values for different actions
- Maximum episode length

### Agent Hyperparameters
Each agent file contains configurable hyperparameters:
- Learning rates
- Network architectures
- Exploration parameters
- Training frequencies

## 📈 Expected Results

### Performance Comparison

**DQN**:
- Expected to show steady improvement through experience replay
- Good final performance but may take longer to converge
- Stable learning curve with occasional plateaus

**PPO**:
- Generally fastest and most stable convergence
- Best balance of exploration and exploitation
- Highest final performance expected

**REINFORCE**:
- More variable learning curve
- May require more episodes to converge
- Simpler but potentially less efficient

### Typical Training Progress
- **Episodes 0-200**: Random exploration, low performance
- **Episodes 200-500**: Learning basic navigation and patient collection
- **Episodes 500-800**: Optimizing drug delivery and room assignments
- **Episodes 800-1000**: Fine-tuning efficiency and policy refinement

## 🐛 Troubleshooting

### Common Issues

1. **Pygame Installation Issues**:
```bash
pip install pygame --upgrade
```

2. **CUDA/GPU Issues**:
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
```

3. **Memory Issues with Large Replay Buffers**:
Reduce buffer size in `dqn_agent.py`:
```python
self.memory = ReplayBuffer(5000)  # Reduced from 10000
```

4. **Slow Training**:
- Reduce episode length: `max_steps=500`
- Use fewer training episodes initially
- Disable rendering during training

## 📚 References

- [Deep Q-Learning Paper](https://arxiv.org/abs/1312.5602)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [REINFORCE Algorithm](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built using Gymnasium framework
- Pygame for visualization
- PyTorch for deep learning implementations
- Inspired by healthcare workflow optimization research