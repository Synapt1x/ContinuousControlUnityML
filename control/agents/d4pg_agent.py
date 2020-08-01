# -*- coding: utf-8 -*-
"""
Custom RL agent for learning how to navigate through the Unity-ML environment
provided in the project. This agent specifically implements the D4PG algorithm.

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
from control.torch_models.actor_net import ActorNetwork
from control.torch_models.critic_net import CriticNetwork
from control.replay_buffer import ReplayBuffer
from control import utils


class D4PGAgent(MainAgent):
    """
    This model contains my code for the agent to learn and be able to interact
    through the continuous control problem.
    """

    def __init__(self, state_size, action_size, num_instances=1, seed=13,
                 **kwargs):
        # first add additional parameters specific to D4PG

        # initialize as in base model
        super(D4PGAgent, self).__init__(state_size, action_size,
                                        num_instances, seed, **kwargs)

    def _init_alg(self):
        """
        Initialize the algorithm based on what algorithm is specified.
        """
        # init storage for actor and critic models
        self.actors = []
        self.actor_targets = []
        self.critics = []
        self.critic_targets = []

        # create all models separately for each agent instance
        for _ in range(self.num_instances):
            actor = ActorNetwork(self.state_size, self.action_size)
            target_actor = ActorNetwork(self.state_size, self.action_size)
            target_actor.load_state_dict(actor.state_dict())

            critic = CriticNetwork(self.state_size, self.action_size)
            target_critic = CriticNetwork(self.state_size, self.action_size)
            target_critic.load_state_dict(critic.state_dict())

            self.actors.append(actor)
            self.actor_targets.append(target_actor)

            self.critics.append(critic)
            self.critic_targets.append(target_critic)

        # initialize the replay buffer
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size,
                                   seed=self.seed)

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

    def compute_loss(self, states, actions, next_states, rewards, dones):
        """
        Compute the loss based on the information provided and the value /
        policy parameterizations used in the algorithm.

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

        #TODO: Implement learning for final algorithm
        return None

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