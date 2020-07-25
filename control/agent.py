# -*- coding: utf-8 -*-
"""
Custom RL agent for learning how to navigate through the Unity-ML environment
provided in the project.

This particularly aims to learn how to solve a continuous control problem.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class MainAgent:
    """
    This model contains my code for the agent to learn and be able to interact
    through the continuous control problem.
    """

    def __init__(self, alg, state_size, action_size, seed=13, **kwargs):
        self.alg = alg
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        np.random.seed(seed)

        # extract hyperparameters for the general algorithm
        self.epsilon = kwargs.get('epsilon', 0.9)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.9999)
        self.epsilon_min = kwargs.get('epsilon_min', 0.05)
        self.gamma = kwargs.get('gamma', 0.9)
        self.alpha = kwargs.get('alpha', 0.2)
        self.t_freq = kwargs.get('t_freq', 10)
        self.tau = kwargs.get('tau', 0.1)

    def _select_random_a(self):
        """
        Select a random action. Actions are clipped to [-1, 1]
        """
        return np.clip(np.random.randint(self.action_size), -1, 1)

    def save_model(self, file_name):
        """
        Save the agent's underlying model(s).

        Parameters
        ----------
        file_name: str
            File name to which the agent will be saved for future use.
        """
        if self.alg.lower() == 'random':
            return None

    def load_model(self, file_name):
        """
        Load the agent's underlying model(s).

        Parameters
        ----------
        file_name: str
            File name from which the agent will be loaded.
        """
        if self.alg.lower() == 'random':
            return None

    def get_action(self, state):
        """
        Extract the action intended by the agent based on the selection
        criteria, either random or using epsilon-greedy policy and taking the
        max from Q(s=state, a) from Q.

        Parameters
        ----------
        state: np.array/torch.Tensor
            Array or Tensor singleton or batch containing state information
            either in the shape (1, 37) or (batch_size, 37)

        Returns
        -------
        int
            Integer indicating the action selected by the agent based on the
            state provided.
        """
        if self.alg.lower() == 'random':
            return self._select_random_a()
        #TODO: Need to implement action selection for final alg

    def compute_update(self, state, action, next_state, reward, done):
        """
        Compute the updated value for the Q-function estimate based on the
        experience tuple.

        Parameters
        ----------
        state: np.array/torch.Tensor
            Array or Tensor singleton or batch containing state information
        action: np.array/torch.Tensor
            Array or Tensor singleton or batch containing actions taken
        next_state: np.array/torch.Tensor
            Array or Tensor singleton or batch containing information about what
            state followed actions taken from the states provided by 'state'
        reward: np.array/torch.Tensor
            Array or Tensor singleton or batch containing reward information
        done: np.array/torch.Tensor
            Array or Tensor singleton or batch representing whether or not the
            episode ended after actions were taken

        Returns
        -------
        torch.float32
            Loss value (with grad) based on target and Q-value estimates.
        """
        if self.alg.lower() == 'random':
            return 0.0
        #TODO: Implement learning for final algorithm

    def learn(self, state, action, next_state, reward, done):
        """
        Learn from an experience tuple. If using DQN, which is the default, then
        store an experience tuple into memory and only learn if enough tuples
        are available in the replay buffer to learn from batches of tuples.

        Parameters
        ----------
        state: np.array/torch.Tensor
            Array or Tensor singleton or batch containing state information
        action: np.array/torch.Tensor
            Array or Tensor singleton or batch containing actions taken
        next_state: np.array/torch.Tensor
            Array or Tensor singleton or batch containing information about what
            state followed actions taken from the states provided by 'state'
        reward: np.array/torch.Tensor
            Array or Tensor singleton or batch containing reward information
        done: np.array/torch.Tensor
            Array or Tensor singleton or batch representing whether or not the
            episode ended after actions were taken
        """
        if self.alg.lower() == 'random':
            return None
        #TODO: Implement learning for final algorithm

    def step(self):
        """
        Update state of the agent and take a step through the learning process
        to reflect experiences have been acquired and/or learned from.
        """
        #TODO: Implement update step for algorithm
        pass