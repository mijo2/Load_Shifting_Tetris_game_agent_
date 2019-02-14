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

LR = 0.01
GAMMA = 0.99

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

env = gym.make('CartPole-v0')

policy = Policy()

policy.load_state_dict(torch.load('/home/mijo/Desktop/Studies/ReinforcementLearning/CartPolev0/model.pt'))

def select_action(state):
    probs = policy(state)
    return probs.max(0)[1].item()

observation = env.reset()
while True:
    env.render()
    action = select_action(torch.Tensor(observation))
    observation, reward, done, info = env.step(action)

    if done:
        break