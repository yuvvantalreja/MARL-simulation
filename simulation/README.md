# Multi-Agent Reinforcement Learning Simulation

A Python-based simulation environment for experimenting with Multi-Agent Reinforcement Learning (MARL) algorithms. This project visualizes how different types of RL agents interact, compete, and cooperate in an environment with limited resources.

## Features

- Interactive simulation with real-time visualization using Pygame
- Multiple agent types:
  - Q-Table based agents (tabular reinforcement learning)
  - Deep Q-Network (DQN) agents with neural network function approximation
- Dynamic resource generation and consumption
- Interactive controls for adjusting simulation parameters
- Real-time metrics and performance visualization
- Emergent cooperative and competitive behaviors

## Requirements

- Python 3.7+
- PyGame
- NumPy
- PyTorch
- Matplotlib (optional, for additional analysis)

## Installation

```bash
# Clone the repository
git clone https://github.com/yuvvantalreja/MARL-simulation.git
cd MARL-simulation

# Set up a virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate

# Install dependencies
pip install pygame numpy torch
```

## Usage

Run the simulation with default parameters:

```bash
python simulation.py
```

## Controls

- **Space**: Pause/Resume simulation
- **R**: Reset simulation
- **ESC**: Quit the application

## Interactive Parameters

The simulation includes interactive sliders to adjust various parameters:

- **Number of Agents**: Controls the agent population
- **Number of Resources**: Sets resource density
- **Learning Rate**: Adjusts how quickly agents learn
- **Exploration Rate**: Controls agent curiosity vs. exploitation
- **Communication Range**: Sets how far agents can communicate
- **Resource Regeneration Rate**: Controls resource replenishment speed

## How It Works

The environment contains agents and resources. Agents move around the environment searching for resources to collect. When an agent collects a resource, it gains energy and reward. 

Agents use reinforcement learning to improve their resource collection strategies over time. The simulation supports different agent types with varying learning mechanisms:

1. **Q-Table Agents**: Use traditional Q-learning with a tabular representation
2. **DQN Agents**: Use deep Q-networks with neural networks for function approximation

The simulation tracks various metrics including average rewards, resource conflicts, cooperative actions, and communication patterns.

## Web Version

A simplified web version of this simulation is also available, built with React and canvas visualization.

## License

MIT

## Author

Yuvvan Talreja
