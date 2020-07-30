# -*- coding: utf-8 -*-
"""
Helper functionality for the continuous control (reacher) task.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import torch


def to_tensor(in_arr, device=torch.device('cpu'), dtype='float'):
    """
    Convert the provided array to a torch tensor.
    """
    tensor = torch.from_numpy(in_arr)

    if dtype == 'float':
        tensor = tensor.float()
    
    return tensor.to(device)