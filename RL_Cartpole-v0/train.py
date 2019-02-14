# ======== IMPORTANT LIBRARIES ======== #

# For using CartPole game environment 
import gym

# Basic libraries
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

# Important Pytorch libraries
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

# ======== MODEL BUILDING ======== #

# HYPERPARAMETERS
LR = 0.01
GAMMA = 0.99

env = gym.make('CartPole-v0')

class Policy(nn.Module):
 def __init__(self):
  super(Policy,self).__init__()

 # Input dimension to neural network, this is 
 # basically the number of values an environment
 # observation gives out. In our case its 4.
  self.state_space = env.observation_space.shape[0]

 # The number of available actions that an agent can
 # choose to perform. Our neural network outputs probabilities
 # corresponding these two actions as in our case the number of 
 # actions is 2.
  self.action_space = env.action_space.n
  

 # A fully connected neural network has been used here. The network
 # consists of 3 layers of which only one is hidden layer
  self.l1 = nn.Linear(self.state_space, 128, bias=False)
  self.l2 = nn.Linear(128, self.action_space, bias=False)
  
  self.gamma = GAMMA

 # The log of the policies stored as a list
  self.policy_history = []
 
 # The cumulative rewards related to different episode numbers
  self.reward_episode = []
  
  self.reward_history = []

 # Losses corresponding to episode numbers
  self.loss_history = []

# The forward function that does the forward propagation
# of the signal and gives probability as output
 def forward(self, x):
  model = torch.nn.Sequential(
      self.l1,
      nn.Dropout(p=0.6),
      nn.ReLU(),
      self.l2,
      nn.Softmax(dim=-1)			 
  )
  return model(x)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=LR)

# This function selects an action(1 or 0) by running policy model and choosing 
# a particular action based on the currently trained policy
def select_action(state,t):
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state))
    c = Categorical(state)
    action = c.sample()
    
    # Logarithm of probability used to store in the policy_history list for loss
    # calculations    
    if len(policy.policy_history) > 0:
        policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action).reshape(1)])
    else:
        policy.policy_history = c.log_prob(action).reshape(1)
    
    return action

# After each episode, the policy is updated by stepping the optimizer
# based on the loss fucntion generated
def update_policy():
  
    # Dummy variable to store discounted reward
    R = 0

    # List that stores Discounted reward corresponding to each timestep
    rewards = []
    
    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0,R)
        
    # We scale the rewards for loss calculations
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    
    policy.policy_history = torch.Tensor(policy.policy_history)
    
    # Loss is calculated as log of probabilities multiplied by corresponding rewards - 1
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))
    
    # Network is updated here and backward propagation is being done
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Reinitialize the history counters and add new loss to the loss_history list
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = []
    policy.reward_episode= []

# The function to episodes 'episodes' number of times
def main(episodes):
    
    # Total reward for all the episodes
    running_reward = 10
    
    for episode in range(episodes):
        state = env.reset() # Reset environment and record the starting state
        
        # The done variable 
        done = False       
         
        # Each episode can run atmost for 1000 timesteps
        for time in range(1000):
          
            # Selecting the action according to state
            action = select_action(state,time)
            
            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.item())

	          # Save reward for each timestep
            policy.reward_episode.append(reward)
            if done:
                break
        
        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)

        # Updating the policy after each episode
        update_policy()
        
        if episode % 50 == 0:
                  print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))

        if running_reward > env.spec.reward_threshold:
                  print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, time))
                  torch.save(policy.state_dict(),'./model.pt')
                  break

# Running 1000 episodes
episodes = 1000
main(episodes)
  