# AI Flappy Bird
A Python implementation of Flappy Bird with both manual play and genetic algorithm AI capabilities.

## Features
- Classic Flappy Bird gameplay
- Genetic algorithm AI that learns to play the game
- Customizable AI parameters
- Support for creating your own AI agents

## Installation

> This project was built and tested using python v3.11.

1. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python -m virtualenv venv

   # Activate on Windows
   .\venv\Scripts\activate

   # Activate on Linux/Mac
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Running the Game

### Manual Play Mode
To play the game yourself:
```bash
python src/main.py --regular
```
Controls:
- Press SPACEBAR to make the bird flap
- Avoid hitting the pipes and the ground
- Try to get through as many pipes as possible!

### AI Mode
To watch the genetic algorithm learn to play:
```bash
python src/main.py
```
This will start the genetic algorithm with:
- Population size: 1000 birds
- Mini-batch size: 1000 birds
- Each generation learns from the best performers of the previous generation

## Creating Your Own AI Agent

The game provides a flexible framework for implementing your own AI agents. Here's how to create one:

1. Create a new file in the `src` directory (e.g., `my_brain.py`)
2. Implement your AI by extending the `BaseGeneticBrain` class:

```python
from brain import BaseGeneticBrain
import numpy as np

class MyCustomBrain(BaseGeneticBrain):
    def __init__(self):
        super().__init__()
        # Add any custom initialization here
        self.weights = np.random.randn(4, 1)  # Example: 4 inputs, 1 output

    def step(self, inputs, keys_pressed):
        """
        inputs contains:
        - distance: horizontal distance to next pipe
        - bird_height: current Y position of bird
        - upper_pipe: Y position of upper pipe's bottom
        - lower_pipe: Y position of lower pipe's top
        """
        # Implement your decision logic here
        ...

    def cross_over_mutation(self, other_brain, num_children):
        """
        Implement breeding logic between two successful brains
        """
        ...

    @staticmethod
    def generate_brains(num_brains: int, epsilon: float, load_latest: bool = False) -> np.array:
        """
        Generate initial population
        """
        return [MyCustomBrain() for _ in range(num_brains)]
```

3. Modify `main.py` to use your custom brain:
```python
from my_brain import MyCustomBrain

def genetic():
    pygame.init()
    window = pygame.display.set_mode(SCREEN_SIZE)
    flock = flappy.FlappyFlock(size=1000, mini_batch=1000, brain_type=MyCustomBrain)
    model = simulator.GeneticGame(flock)
    display = simulator.GeneticGameBoard(SCREEN_SIZE[0], True)
    sim = simulator.Simulator(model, display, window, fps=FPS)
    sim.start()
```

## How the Genetic Algorithm Works

1. **Initialization**: Creates a population of birds with random neural networks
2. **Evaluation**: Each bird attempts to play the game
3. **Selection**: The best performing birds (highest scores) are selected
4. **Breeding**: Selected birds' neural networks are combined to create new birds
5. **Mutation**: Random changes are applied to maintain diversity
6. **Repeat**: The process continues with the new generation

## Additional Challenge

You can also extend the `BaseBrain` class to implement non-genetic algorithm approaches. Here are some interesting ideas to try:

1. **Q-Learning Agent**
   - Implement a Q-learning algorithm that learns optimal actions for different game states
   - Use discretized state spaces for bird height and pipe positions
   - Experiment with different reward structures and learning rates

2. **Rule-Based System**
   - Create a simple but effective agent using hand-crafted rules
   - Consider factors like optimal flying height and safe zones
   - Challenge yourself to achieve a high score with minimal complexity

3. **Deep Reinforcement Learning**
   - Implement a DQN (Deep Q-Network) agent
   - Use PyTorch or TensorFlow to create a neural network
   - Train the agent using experience replay

4. **Monte Carlo Tree Search**
   - Implement MCTS to look ahead and plan optimal moves
   - Use simulation rollouts to evaluate different actions
   - Balance exploration vs exploitation

Example implementation of a simple rule-based agent:

```python
from brain import BaseBrain

class RuleBasedBrain(BaseBrain):
    def __init__(self):
        super().__init__()

    def step(self, inputs, keys_pressed):
        ...
```

Choose any of these approaches and create your own implementation by extending `BaseBrain`. Compare your results against the genetic algorithm solution!

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


