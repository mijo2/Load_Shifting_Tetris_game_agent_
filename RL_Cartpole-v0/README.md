# CartPole Version 0
This directory contains files that create and train an agent that plays the game CartPole version 0

## Introduction 

### Details of the game
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

### Actions
The agent/player can take two actions namely:
  * Left(0)
  * Right(1)

### Task 
The task of the agent/player is fairly simple. It is to play for as many timesteps as possible thus getting a high cumulative reward.


## Solution

The solution used to make this agent is relatively simple.

### Input
The observation list that gym emits at each timestep is used as to give an input to the agent network for training as well as testing. 
The observation or rather the state of the environment looks something like below:
  [position of cart, velocity of cart, angle of pole, rotation rate of pole]

### Network

The agent network is a 3-layer fully connected neural network with only one hidden layer.
![alt text](https://github.com/mijo2/Tetris_game_agent/blob/master/RL_Cartpole-v0/Images/nn.jpg)

The number of neurons in:  
    1. Input Layer: 4  
    2. Hidden Layer: 128  
    3. Output Layer: 2  





