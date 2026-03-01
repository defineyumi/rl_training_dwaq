# Robust Quadrupedal Locomotion on Complex Terrains via Adaptive Entropy Learning

This repository contains the implementation of a reinforcement learning-based locomotion control system for quadruped robots, featuring adaptive entropy learning and VHIP-based stability rewards. The code supports training and deployment on DeepRobotics Lite3 and X30 quadruped robots.

## 🎬 Demo

![Quadruped Robot Locomotion Demo](issacgym_test.gif)

## ✨ Key Features

- **CENet-based Terrain Imagination**: Implicit terrain feature learning using Context-Encoded Network (CENet) from DreamWaQ framework
- **VHIP-based Stability Rewards**: Dynamic stability constraints using Variable Height Inverted Pendulum (VHIP) model
- **Dynamic Entropy Coefficient**: Adaptive entropy adjustment mechanism inspired by Simulated Annealing for efficient exploration-exploitation balance
- **Curriculum Learning**: Progressive terrain difficulty and velocity command curriculum
- **Domain Randomization**: Comprehensive randomization for robust sim-to-real transfer
- **Real-world Deployment**: Validated on DeepRobotics Lite3 robot in various challenging terrains

## 🎯 Performance

- **Training Efficiency**: Achieves successful locomotion after ~500 iterations (~200M samples)
- **Final Performance**: Reaches terrain level 6.0 ± 0.03 after 4000 iterations
- **Velocity Tracking**: Linear velocity tracking accuracy of 0.78 ± 0.015
- **Stability**: Significantly reduced fall rates on complex terrains (2% at level 5, 3% at level 6 with VHIP rewards)

## 🔧 Requirements

We test our codes under the following environment:

- Ubuntu 20.04
- NVIDIA Driver: 525.147.05
- CUDA 12.0
- Python 3.7.16
- PyTorch 1.10.0+cu113
- Isaac Gym: Preview 4

**Hardware Requirements:**
- GPU: NVIDIA GPU with at least 16GB VRAM (tested on RTX 4070Ti)
- RAM: 16GB+ recommended

## 📦 Installation

### 1. Create Environment and Install PyTorch

```bash
conda create -n ppo python=3.7.16
conda activate ppo
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

### 2. Install Isaac Gym

1. Download Isaac Gym Preview 4 from [NVIDIA Developer](https://developer.nvidia.com/isaac-gym)
2. Extract and install:
```bash
cd isaacgym/python
pip install -e .
```

### 3. Clone Repository

```bash
git clone https://github.com/defineyumi/rl_training_dwaq.git
cd rl_training
```

### 4. Install Dependencies

```bash
# Install rsl_rl (PPO implementation with modifications)
cd rsl_rl
pip install -e .

# Install legged_gym (simulation environment with modifications)
cd ../legged_gym
pip install -e .
```

**Note:** Please use the `legged_gym` and `rsl_rl` provided in this repository, as we have made modifications to these repos for our implementation.

## 🚀 Usage

### Training

To train a policy from scratch:

```bash
cd legged_gym/legged_gym/scripts
python train.py
```

**Training Configuration:**
- Default: 4096 parallel environments
- Training iterations: 4000
- Samples per iteration: 409,600
- Total training time: ~4 hours on RTX 4070Ti

**Key Hyperparameters:**
- PPO clip range: 0.2
- GAE factor (λ): 0.95
- Discount factor (γ): 0.99
- Learning rate: 1×10⁻³
- Dynamic entropy coefficient range: [0.01, 0.02]

### Testing/Evaluation

To test and export the latest trained policy:

```bash
cd legged_gym/legged_gym/scripts
python play.py
```

### Real-world Deployment

For deployment on DeepRobotics Lite3 robot:

1. Export the trained policy from simulation
2. Deploy to robot's onboard computer
3. Control frequency: 50Hz
4. PD controller gains: Kp=40, Kd=1

## 📁 Project Structure

```
rl_training_dwaq/
├── legged_gym/              # Simulation environment
│   ├── legged_gym/
│   │   ├── envs/
│   │   │   ├── lite3/       # Lite3 robot configuration
│   │   │   │   └── lite3_Dwaq_config.py  # Our modified config
│   │   │   └── x30/         # X30 robot configuration
│   │   └── scripts/
│   │       ├── train.py      # Training script
│   │       └── play.py       # Testing/evaluation script
│   └── resources/
│       └── robots/          # Robot URDF files and meshes
├── rsl_rl/                   # PPO implementation
│   └── rsl_rl/
│       ├── algorithms/
│       │   └── ppo.py        # PPO with dynamic entropy coefficient
│       └── modules/
│           └── actor_critic.py  # Actor-critic networks
└── README.md
```

## 🔬 Key Components

### 1. CENet (Context-Encoded Network)
- Encodes historical observations (5 timesteps) into latent terrain features
- Provides velocity estimation and terrain feature extraction
- Input: 270-dim (6 timesteps × 45 dims)
- Output: 19-dim encoding (3-dim velocity + 16-dim latent features)

### 2. VHIP-based Rewards
- VHIP angle penalty: penalizes excessive torso tilt (threshold: 0.1 rad)
- VHIP angular acceleration penalty: penalizes rapid balance loss (threshold: 0.01)
- Stand still pose reward: maintains stable standing posture at low velocities

### 3. Dynamic Entropy Coefficient
- Adaptively adjusts based on performance metrics (linear velocity, angular velocity, terrain utilization)
- Range: [0.01, 0.02]
- Enables efficient exploration in early training and stable exploitation in later stages

## 📊 Experimental Results

### Simulation Results
- **Baseline Comparison**: Outperforms DreamWaQ in final terrain level (6.0 vs. 5.9) and convergence speed (3000 vs. 3500 iterations to reach level 6)
- **Ablation Studies**: VHIP rewards reduce fall rates from 5% to 3% at terrain level 6
- **Training Efficiency**: Achieves locomotion capability in ~500 iterations

### Real-world Validation
Successfully deployed on DeepRobotics Lite3 robot in various terrains:
- Stairs (16cm step height)
- Rough surfaces (woven bags)
- Smooth surfaces (marble floors)
- Natural terrains (bushes, deep grassland)

## 📝 License

Please check the license files in `legged_gym/LICENSE` and `rsl_rl/LICENSE` for details.

## 🔗 Related Work

This implementation is based on:
- **DreamWaQ**: Terrain imagination framework (CENet)
- **Isaac Gym**: High-performance GPU-based physics simulation
- **RSL-RL**: Reinforcement learning library for legged robots

## 📚 References

- DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning (ICRA 2023)
- Isaac Gym: High Performance GPU-Based Physics Simulation for Robot Learning (NeurIPS 2021)

---

**Note**: This code is provided for research purposes. Please ensure you comply with all applicable licenses and regulations when using this software.
