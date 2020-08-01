# -*- coding: utf-8 -*-
"""
Main code for the continuous control (reacher) task.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import os
import argparse
import datetime
import pickle
import time

from unityagents import UnityEnvironment
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use backend for saving plots only
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from control import utils


class ControlMain:
    """
    This code contains functionality for running the Reacher environment and
    running the code as per training, showing performance, and loading/saving my
    models.
    """

    def __init__(self, file_path, alg, model_params, frame_time=0.075,
                 max_episodes=1E5, max_iterations=1E5):
        self.frame_time = frame_time
        self.max_iterations = max_iterations
        self.max_episodes = max_episodes

        self.env, self.brain_name, self.brain = self._init_env(file_path)
        self.agent = self._init_agent(alg, model_params)

        self.score_store = []
        self.average_scores = []

    def _init_env(self, file_path):
        """
        Initialize the Unity-ML Reacher environment.
        """
        env = UnityEnvironment(file_name=file_path)
        brain_name = env.brain_names[0]
        first_brain = env.brains[brain_name]

        return env, brain_name, first_brain

    def _init_agent(self, alg, model_params):
        """
        Initialize the custom model utilized by the agent.
        """
        # extract state and action information
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        num_agents = len(env_info.agents)
        state_size = len(env_info.vector_observations[0])
        action_size = self.brain.vector_action_space_size

        if alg.lower() == 'ddpg':
            # init DDPG
            from control.agents.ddpg_agent import DDPGAgent
            return DDPGAgent(**model_params, state_size=state_size,
                             action_size=action_size, num_instances=num_agents)
        elif alg.lower() == 'd4pg':
            # init D4PG
            from control.agents.d4pg_agent import D4PGAgent
            return D4PGAgent(**model_params, state_size=state_size,
                             action_size=action_size, num_instances=num_agents)
        else:
            # default to random
            from control.agents.agent import MainAgent
            return MainAgent(**model_params, state_size=state_size,
                            action_size=action_size, num_instances=num_agents)

    def _update_scores(self, scores):
        """
        Store scores from an episode into running storage.
        """
        # store average over agents
        self.score_store.append(np.mean(scores))

        # also store average over last 100 episodes over agent average
        score_avg = np.mean(self.score_store[-100:])
        self.average_scores.append(score_avg)

    def save_model(self, file_name):
        """
        Save the model to the file name specified.

        Parameters
        ----------
        file_name: str
            File name to which the agent will be saved for future use.
        """
        self.agent.save_model(file_name)

    def load_model(self, file_name):
        """
        Load the model specified.

        Parameters
        ----------
        file_name: str
            File name from which the agent will be loaded.
        """
        self.agent.load_model(file_name)

    def save_training_plot(self, first_solved):
        """
        Plot training performance through episodes.

        Parameters
        ----------
        first_solved: int
            Episode number at which the agent solved the continuous control
            problem by achieving an average score of +13.
        """
        #TODO: Needs to be updated to new plotting
        num_eval = len(self.average_scores)

        if num_eval > 100:
            # Set up plot file and directory names
            out_dir, cur_date = utlis.get_output_dir()
            plot_file = os.path.join(out_dir,
                                     f'training-performance-{cur_date}.png')

            # plot and save the plot file
            fig = plt.figure(figsize=(12, 8))

            plt.plot(self.score_store, linewidth=1, alpha=0.4,
                     label='raw_episode_score')
            plt.plot(self.average_scores, linewidth=2,
                     label='100_episode_avg_score')
            plt.title(f'Average Score Over Recent 100 Episodes During Training',
                      fontsize=20)

            plt.xlabel('Episode', fontsize=16)
            plt.ylabel('Average Score', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt.xlim([0, num_eval])
            plt.ylim([0, np.max(self.score_store)])

            # plot indicator for solved iteration
            if first_solved > 0:
                min_val = np.min(self.average_scores)
                plt.axhline(y=13, color='g', linewidth=1, linestyle='--')
                ax = fig.gca()

                ax.add_artist(Ellipse((first_solved, 13),
                                      width=20, height=0.3, facecolor='None',
                                      edgecolor='r', linewidth=3, zorder=10))
                plt.text(first_solved + 10, 12.25,
                         f'Solved in {first_solved} episodes', color='r',
                         fontsize=14)

            plt.legend(fontsize=12)

            plt.savefig(plot_file)

            print(f'Training plot saved to {plot_file}')
        else:
            print('Not enough average scores computed. Skipping plotting.')

    def save_results(self):
        """
        Save training averages over time.
        """
        #TODO: Needs to be updated slightly for new results
        num_eval = len(self.average_scores)

        if num_eval > 100:
            # Save results
            out_dir, cur_date = utils.get_output_dir()
            res_file = os.path.join(out_dir,
                                    f'results-file-{cur_date}.pkl')

            with open(res_file, 'wb') as o_file:
                pickle.dump(self.average_scores, o_file)

            print(f'Training results saved to {res_file}')
        else:
            print('Not enough average score computed. Skipping saving results.')

    def run_episode(self, train_mode=True):
        """
        Run an episode of interaction in the Unity-ML Reacher environment.

        Parameters
        ----------
        train_mode: bool
            Flag to indicate whether or not the agent will be training or just
            running inference.
        """
        iteration = 0

        # initiate interaction and learning in environment
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        states = env_info.vector_observations

        # get the number of agents and initialize a score for each
        scores = np.zeros(self.agent.num_instances)

        while iteration < self.max_iterations:
            # first have the agent act and evaluate state
            actions = self.agent.get_action(utils.to_tensor(states))
            env_info = self.env.step(actions)[self.brain_name]
            next_states, rewards, dones = utils.eval_state(env_info)

            # learn from experience tuple batch
            if train_mode:
                self.agent.learn(states, actions, next_states, rewards, dones)

            # increment score and compute average
            scores += rewards
            states = next_states

            if np.any(np.array(dones)):
                break
            time.sleep(self.frame_time)

            # print average score as training progresses
            iteration += 1

        return scores

    def train_agent(self, train_mode=True):
        """
        Train an agent by running learning episodes in the Reacher task.

        Parameters
        ----------
        train_mode: bool
            Flag to indicate whether or not the agent will be training or just
            running inference.
        """
        episode = 1
        try:
            #TODO: Needs to be updated for parallel training
            # run episodes
            while episode < self.max_episodes:
                scores = self.run_episode(train_mode=train_mode)

                self._update_scores(scores)

                print(f'* Episode {episode} completed * avg: {np.mean(scores)} *')

                episode += 1
                if train_mode:
                    self.agent.step()

        except KeyboardInterrupt:
            print("Exiting learning gracefully...")
        finally:
            self.env.close()
