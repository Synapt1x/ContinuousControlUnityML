# -*- coding: utf-8 -*-
"""
This is a custom dataset is for holding batches of trajectories. Specific use
is for PPO.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Custom dataset for loading batch data training the PPO algorithm.
    """

    def __init__(self, states, actions, next_states, rewards, dones):
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.rewards = rewards
        self.dones = dones

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return tuple([self.states[idx],
                     self.actions[idx],
                     self.next_states[idx],
                     self.rewards[idx],
                     self.dones[idx]])