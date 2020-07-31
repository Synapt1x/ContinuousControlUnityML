# -*- coding: utf-8 -*-
"""
Helper functionality for the continuous control (reacher) task.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import os
import datetime

import numpy as np
import torch


def eval_state(curr_env_info):
    """
    Evaluate a provided game state.
    """
    s = curr_env_info.vector_observations
    r = curr_env_info.rewards
    d = np.array(curr_env_info.local_done).astype(int)  # convert bool->int

    return s, r, d


def print_progress(iteration, score_avg):
    """
    Helper method for printing out the state of the game after completion.
    """
    print(f"Average score so far: {score_avg}")


def print_on_close(score):
    """
    Helper method for printing out the state of the game after completion.
    """
    print(f"Final Score: {score}")


def get_output_dir():
    """
    Return the output file path.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cur_dir, os.pardir, 'output')
    cur_date = datetime.datetime.now().strftime('%Y-%m-%d')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    return out_dir, cur_date


def to_tensor(in_arr, device=torch.device('cpu'), dtype='float'):
    """
    Convert the provided array to a torch tensor.
    """
    tensor = torch.from_numpy(in_arr)

    if dtype == 'float':
        tensor = tensor.float()

    return tensor.to(device)