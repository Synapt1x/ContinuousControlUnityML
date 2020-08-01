# -*- coding: utf-8 -*-
"""
Simple torch model that is a set of fully connected layers.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import numpy as np
import torch
from torch import nn


class ActorNetwork(nn.Module):
    """
    Torch model containing a set of dense fully connected layers.
    """

    def __init__(self, state_size, action_size, inter_dims=None, seed=13):
        if inter_dims is None:
            self.inter_dims = [64, 256]
        else:
            self.inter_dims = inter_dims

        self.state_size = state_size
        self.action_size = action_size

        # set the seed
        self.seed = seed
        torch.manual_seed(self.seed)

        super(ActorNetwork, self).__init__()

        # initialize the architecture
        self._init_model()

    def _init_model(self):
        """
        Define the architecture and all layers in the model.
        """
        self.state_batch = nn.BatchNorm1d(self.state_size)
        self.input = nn.Linear(self.state_size, self.inter_dims[0])
        self.input_batch = nn.BatchNorm1d(self.inter_dims[0])

        hidden_layers = []
        batch_norms = []

        for dim_i, hidden_dim in enumerate(self.inter_dims[1:]):
            prev_dim = self.inter_dims[dim_i]
            hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.hidden_batch_norms = nn.ModuleList(batch_norms)
        self.output = nn.Linear(self.inter_dims[-1], self.action_size)

    def forward(self, state):
        """
        Define the forward-pass for data through the model.

        Parameters
        ----------
        state: torch.Tensor
            A 37-length Torch.Tensor containing a state vector to be run through
            the network.

        Returns
        -------
        torch.Tensor
            Tensor containing output action values determined by the network.
        """
        data_x = self.state_batch(state.float())
        data_x = self.input_batch(torch.relu(self.input(data_x)))
        for layer, batch_norm in zip(self.hidden_layers,
                                     self.hidden_batch_norms):
            data_x = batch_norm(torch.relu(layer(data_x)))
        action_values = torch.tanh(self.output(data_x))

        return action_values