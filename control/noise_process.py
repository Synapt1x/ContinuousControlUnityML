# -*- coding: utf-8 -*-
"""
This code contains my implementation of an Ornstein-Uhlenbeck process for
generating random noise during action selection. This implementation was
inspired by and borrows from:

https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import numpy as np


class OrnsteinUhlenbeck():
    """
    This class contains functionality for sampling from a random noise process
    in order to add exploratory noise to actions during the DDPG algorithm.
    Since the original authors in Lillicrap et al. utilized this stochastic
    process for noise, I aimed to test this as well starting from a base
    implementation provided by:

    https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method

    This code specifically samples from the process using the Euler-Maruyama
    method for sampling by discretizing as:

    dY_t = theta * (mu - Y_t) * dt + sigma * d W_t

    using base Wiener noise W_t (Gaussian noise with time-bound decay).
    """

    def __init__(self, dt, theta=0.15, mu=0.0, sigma=0.2, seed=13):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.seed = seed

        # seed only on construction not every reset
        np.random.seed(self.seed)

        self.init_process()

    def init_process(self):
        """
        Initialize or reset the Ornstein-Uhlenbeck process.
        """
        self.y = 0.0

    def uhlenbeck_mu(self):
        """
        Calculate the Ornstein-Uhlenbeck mu based on theta * (mu - Y_t)
        """
        return self.theta * (self.mu - self.y)

    def dw(self):
        """
        Sample the Wiener process noise grad d W_t.
        """
        return np.random.randn() * np.sqrt(self.dt)

    def sample(self):
        """
        Extract a sample at time step t. This will provide a single sample of
        noise that will continually mean-revert to mu until reset with
        init_process().

        This essentially computes:
            Y_t+1 = Y_t + dY_t
            where
            dY_t = theta * (mu - Y_t) * dt + sigma * d W_t
        """
        y = self.y + self.uhlenbeck_mu() * self.dt + self.sigma * self.dw()
        self.y = y

        return y

if __name__ == '__main__':

    from matplotlib import pyplot as plt

    print('Testing Ornstein-Uhlenbeck process noise and saving to '
          'test_uhlenbeck.png')

    t_end = 1
    t_start = 0
    n = 10000
    dt = (t_end - t_start ) / n
    t = np.linspace(t_end, t_start, n)
    y = []

    uo = OrnsteinUhlenbeck(dt=dt)
    for i in range(n):
        y.append(uo.sample())

    plt.plot(t, y)
    plt.savefig('test_uhlenbeck.png')