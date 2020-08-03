# -*- coding: utf-8 -*-
"""
This wrapper CLI enables defining and running experiments to crudely optimize
parameters for the underlying algorithm being testing.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import os
import datetime
import yaml
import argparse

from control.control_main import ControlMain
from runner import load_config, parse_config, main as run_model


# global constants
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(CUR_DIR, 'configs', 'default_config.yaml')

################### Experimental Parameters ########################
exp_params = {'alpha': [0.0001, 0.001, 0.01, 0.05]}
####################################################################


def parse_args():
    """
    Parse provided arguments from the command line.
    """
    arg_parser = argparse.ArgumentParser(
        description="Argument parsing for accessing andrunning my deep "\
            "reinforcement learning projects")

    # command-line arguments
    arg_parser.add_argument('-c', '--config', dest='config_file',
                            type=str, default=DEFAULT_CONFIG)
    args = vars(arg_parser.parse_args())

    return args


def main(config_file=DEFAULT_CONFIG):
    """
    Main runner for the code CLI.
    """
    args = parse_args()

    config_file = args.get('config_file', DEFAULT_CONFIG)
    config_args = load_config(config_file)
    orig_config_data, model_file, model_params = parse_config(config_args)

    # run model separately for each parameter
    for param, values in exp_params.items():
        for p_val in values:
            model_params[param] = p_val

            # update the output graph file names
            value_str = str(p_val).replace('.', '-')
            g_name = orig_config_data['graph_file'].replace(
                '.png', f'-{param}-{value_str}.png')
            config_data = orig_config_data.copy()
            config_data['graph_file'] = g_name

            # run the model with the current provided parameters
            run_model(model_file, model_params, train=True,
                      config_data=config_data)


if __name__ == '__main__':
    args = parse_args()
    main(**args)
