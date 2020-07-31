# -*- coding: utf-8 -*-
"""
Custom RL agent for learning how to navigate through the Unity-ML environment
provided in the project. This agent specifically implements the DDPG algorithm.

This particularly aims to learn how to solve a continuous control problem.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from control.agents.agent import MainAgent
from control.torch_models.simple_linear import LinearModel
from control.replay_buffer import ReplayBuffer
from control import utils


class DDPGAgent(MainAgent):
    """
    This model contains my code for the agent to learn and be able to interact
    through the continuous control problem.
    """

    def __init__(self, state_size, action_size, num_instances=1, seed=13,
                 **kwargs):
        # first additional parameters specific to DDPG

        # initialize as in base model
        super(DDPGAgent, self).__init__(state_size, action_size,
                                        num_instances, seed, **kwargs)

    def _init_alg(self):
        """
        Initialize the algorithm based on what algorithm is specified.
        """
        #TODO: Implement final algorithm initialization for DDPG
        raise ValueError('Main model called rather than specific algorithm.')

    def save_model(self, file_name):
        """
        Save the agent's underlying model(s).

        Parameters
        ----------
        file_name: str
            File name to which the agent will be saved for future use.
        """
        return None

    def load_model(self, file_name):
        """
        Load the agent's underlying model(s).

        Parameters
        ----------
        file_name: str
            File name from which the agent will be loaded.
        """
        return None

    def get_action(self, states, in_train=True):
        """
        Extract the action intended by the agent based on the selection
        criteria, either random or using epsilon-greedy policy and taking the
        max from Q(s=state, a) from Q.

        Parameters
        ----------
        states: np.array/torch.Tensor
            Array or Tensor singleton or batch containing states information
            either in the shape (1, 37) or (batch_size, 37)

        Returns
        -------
        int
            Integer indicating the action selected by the agent based on the
            states provided.
        """
        #TODO: Implement action selection for final algorithm
        return self._select_random_a()

    def compute_update(self, states, actions, next_states, rewards, dones):
        """
        Compute the updated value for the Q-function estimate based on the
        experience tuple.

        Parameters
        ----------
        states: np.array/torch.Tensor
            Array or Tensor singleton or batch containing states information
        actions: np.array/torch.Tensor
            Array or Tensor singleton or batch containing actions taken
        next_states: np.array/torch.Tensor
            Array or Tensor singleton or batch containing information about what
            state followed actions taken from the states provided by 'state'
        rewards: np.array/torch.Tensor
            Array or Tensor singleton or batch containing reward information
        dones: np.array/torch.Tensor
            Array or Tensor singleton or batch representing whether or not the
            episode ended after actions were taken

        Returns
        -------
        torch.float32
            Loss value (with grad) based on target and Q-value estimates.
        """
        #TODO: Implement computation for final algorithm
        return 0.0

    def learn(self, states, actions, next_states, rewards, dones):
        """
        Learn from an experience tuple.

        Parameters
        ----------
        states: np.array/torch.Tensor
            Array or Tensor singleton or batch containing states information
        actions: np.array/torch.Tensor
            Array or Tensor singleton or batch containing actions taken
        next_states: np.array/torch.Tensor
            Array or Tensor singleton or batch containing information about what
            state followed actions taken from the states provided by 'state'
        rewards: np.array/torch.Tensor
            Array or Tensor singleton or batch containing reward information
        dones: np.array/torch.Tensor
            Array or Tensor singleton or batch representing whether or not the
            episode ended after actions were taken
        """
        #TODO: Implement training for algorithm
        return None

    def step(self):
        """
        Update state of the agent and take a step through the learning process
        to reflect experiences have been acquired and/or learned from.
        """
        #TODO: Implement update step for algorithm
        return None