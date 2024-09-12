# Blackjack Reinforcement Learning
This repository contains an implementation of a Monte Carlo Control algorithm for solving the Blackjack game using Reinforcement Learning (RL). The RL agent learns an optimal policy to play Blackjack by interacting with a custom environment based on the Gymnasium library.

## Overview
The code includes:
1. A custom Blackjack environment derived from the Gymnasium library.
2. Monte Carlo Control methods for policy evaluation and improvement.
3. Visualization of the learned policy and value functions using Matplotlib.
4. A graphical user interface (GUI) built with Tkinter to display the plots.


## Components
### Custom Environment
The CustomBlackjackEnv class extends the BlackjackEnv from Gymnasium to customize the observation and action spaces. This environment simulates the Blackjack game, where the player can either “hit” or “stick” and receives rewards based on the outcome of the game.

### Monte Carlo Control
The Monte Carlo Control algorithm is implemented to find the optimal policy for the Blackjack game. The key functions include:
- generate_episode_from_Q(): Generates episodes based on the current Q-values and policy.
- get_probs(): Computes action probabilities for a given state using an epsilon-greedy strategy.
- update_Q(): Updates the Q-values based on the episode’s rewards and states.
- mc_control(): Runs the Monte Carlo Control algorithm over a specified number of episodes to learn the optimal policy.

### Visualization
The BlackjackGUI class provides methods to visualize:
- The optimal policy as a matrix where the rows represent the player’s hand values and columns represent the dealer’s visible card.
- The expected return for different states, showcasing how the agent values different states under various conditions.

### Running the Code
To run the code, execute the script in your Python environment. The script will:
1. Create a custom Blackjack environment.
2. Train the Monte Carlo Control algorithm over 500,000 episodes with an alpha value of 0.02.
3. Display the optimal policy and value plots in separate windows.

### Configuration
You can customize the number of episodes and the alpha rate by modifying the following parameters in the mc_control function call:
- num_episodes: Set this to the desired number of episodes for training.
- alpha: Adjust the learning rate (alpha) to control how much the Q-values are updated.

## Example Usage
To change the number of episodes or the alpha rate, update the mc_control function call in the __main__ block:
```commandline
policy, Q = mc_control(env, num_episodes=1_000_000, alpha=0.01)
```
This will run the Monte Carlo Control with 1,000,000 episodes and a learning rate of 0.01.

### Notice
Feel free to experiment with the number of episodes and alpha values to see how they affect the learned policy and value functions.


## Requirements
- numpy
- matplotlib
- gymnasium
- tkinter (usually included with Python)

 Enjoy exploring the Blackjack game with Reinforcement Learning!
