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

from control.replay_buffer import ReplayBuffer
from control import utils

class MainAgent:
    """
    This model contains my code for the agent to learn and be able to interact
    through the continuous control problem.
    """

    def __init__(self, state_size, action_size, num_instances=1, seed=13,
                 **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.num_instances = num_instances  # number of parallel agents
        self.seed = seed

        self.critic_loss_avgs = []
        self.actor_loss_avgs = []

        np.random.seed(seed)

        # initalize device; use GPU if available
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # set the seed for torch models as well
        torch.manual_seed(seed)
        if self.device.type != 'cpu':
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # extract hyperparameters for the general algorithm
        self.gamma = kwargs.get('gamma', 0.9)
        self.actor_alpha = kwargs.get('actor_alpha', 0.0001)
        self.critic_alpha = kwargs.get('critic_alpha', 0.001)
        self.t_freq = kwargs.get('t_freq', 10)
        self.tau = kwargs.get('tau', 0.1)
        self.actor_inter_dims = kwargs.get('actor_inter_dims', [256, 256])
        self.critic_inter_dims = kwargs.get('critic_inter_dims',
                                            [128, 256, 128])
        self.use_batch_norm = kwargs.get('use_batch_norm', False)

        # extract parameters specific to replay buffer
        self.buffer_size = kwargs.get('buffer_size', 1E6)
        self.batch_size = kwargs.get('batch_size', 32)

        # initialize time step and update step parameters
        self.t = 0
        self.t_update = kwargs.get('t_update', 20)
        self.num_updates = kwargs.get('num_updates', 10)
        self.tau = self.tau / self.num_updates

        # initialize random noise parameters
        self.use_ornstein = kwargs.get('use_ornstein', True)
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.99)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        # initialize parameters for Ornstein
        self.theta = kwargs.get('theta', 0.15)
        self.sigma = kwargs.get('sigma', 0.20)
        self.decay = kwargs.get('decay', 0.9996)
        noise_variance = kwargs.get('noise_variance', 0.3)

        self._init_alg()
        self.noise = self._init_noise(noise_variance)

    def _init_alg(self):
        """
        Initialize the algorithm.
        """
        return None

    def _init_noise(self, noise_variance):
        """
        Get the noise process specified.
        """
        if self.use_ornstein:
            from control.noise_processes.noise_process import OrnsteinUhlenbeck

            return OrnsteinUhlenbeck(theta=self.theta, sigma=self.sigma,
                                     action_size=self.action_size,
                                     decay=self.decay)
        else:
            from control.noise_processes.normal_noise import NormalNoise

            return NormalNoise(epsilon=self.epsilon,
                               epsilon_decay=self.epsilon_decay,
                               epsilon_min=self.epsilon_min,
                               noise_variance=noise_variance)

    def _select_random_a(self):
        """
        Select action probabilities randomly. Action probs are clipped to
        [-1, 1].
        """
        actions = np.random.randn(self.num_instances, self.action_size)
        actions = torch.from_numpy(actions)
        return torch.clamp(actions, -1, 1)

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
        return self._select_random_a()

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
            Array or Tensor singleton or batch containing information about
            what state followed actions taken from the states provided
        rewards: np.array/torch.Tensor
            Array or Tensor singleton or batch containing reward information
        dones: np.array/torch.Tensor
            Array or Tensor singleton or batch representing whether or not the
            episode ended after actions were taken
        """
        return None

    def step(self):
        """
        Update state of the agent and take a step through the learning process
        to reflect experiences have been acquired and/or learned from.
        """
        return None

    def decay_epsilon(self):
        """
        Decay the agents epsilon for multiplying added action noise.
        """
        # decay epsilon for random noise
        self.epsilon = np.max([self.epsilon * self.epsilon_decay,
                               self.epsilon_min])