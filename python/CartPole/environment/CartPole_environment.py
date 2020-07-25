#!/usr/bin/python
# -*- coding: utf-8 -*-

from python.RLutils.environment.gymEnvironment import GymEnvironment

class CartPoleEnvironment(GymEnvironment):

    def __init__(self, hyperparams):
        env_name = 'CartPole-v0'
        super(CartPoleEnvironment, self).__init__(hyperparams, env_name)
        return

if __name__ == "__main__":
    cartpole_env = CartPoleEnvironment({})
