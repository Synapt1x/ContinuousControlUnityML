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

from control.torch_models.simple_linear import LinearModel
from control.replay_buffer import ReplayBuffer
from control import utils

class MainAgent:
    """
    This model contains my code for the agent to learn and be able to interact
    through the continuous control problem.
    """

    def __init__(self, alg, state_size, action_size, num_instances=1, seed=13,
                 **kwargs):
        self.alg = alg
        self.state_size = state_size
        self.action_size = action_size
        self.num_instances = num_instances  # number of parallel agents
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

        # parameters for the replay buffer
        self.buffer_size = kwargs.get('buffer_size', 1E6)
        self.batch_size = kwargs.get('batch_size', 32)

        # init what will need to be defined for D4PG
        self.actors = []
        self.actor_targets = []
        self.critics = []
        self.critic_targets = []

        # create all models separately for each agent instance
        for _ in range(num_instances):
            for model_type in range(2):
                base_model = LinearModel(self.state_size, self.action_size)
                target_model = LinearModel(self.state_size, self.action_size)
                target_model.load_state_dict(base_model.state_dict())

                if model_type == 0:
                    self.actors.append(base_model)
                    self.actor_targets.append(target_model)
                else:
                    self.critics.append(base_model)
                    self.critic_targets.append(target_model)

        # initialize the replay buffer
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size,
                                   seed=seed)

    def _select_random_a(self):
        """
        Select action probabilities randomly. Action probs are clipped to
        [-1, 1].
        """
        actions = np.random.randn(self.num_instances, self.action_size)
        return np.clip(actions, -1, 1)

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
        if self.alg.lower() == 'random':
            return self._select_random_a()

        # return action as actor policy + random noise for exploration
        actions = [[]] * self.num_instances
        for agent_num in range(self.num_instances):
            noise = self.epsilon * torch.randn(1, self.action_size)  #TODO: device
            actor = self.actors[agent_num]

            # compute actions for this agent after detaching from training
            actor.eval()
            with torch.no_grad():
                state_vals = states[agent_num]
                action_values = self.actors[agent_num](state_vals) + noise
                actions[agent_num] = torch.clamp(
                    action_values.squeeze(0), -1, 1)
            
            if in_train:
                actor.train()
        actions = torch.stack(actions)

        return actions.numpy()

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
        if self.alg.lower() == 'random':
            return 0.0
        #TODO: Implement learning for final algorithm

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
        if self.alg.lower() == 'random':
            return None

        # first append all agent data to replay buffer
        self.memory.store_tuple(states, actions, rewards, next_states,
                                dones)

        if len(self.memory) >= self.batch_size:
            exp_tuples = self.memory.sample()
            states, actions, rewards, next_states, dones = exp_tuples

            #TODO: finish

    def step(self):
        """
        Update state of the agent and take a step through the learning process
        to reflect experiences have been acquired and/or learned from.
        """
        #TODO: Implement update step for algorithm
        pass