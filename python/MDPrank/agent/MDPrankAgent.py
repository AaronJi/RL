#!/usr/bin/python
# -*- coding: <encoding name> -*-

import os
import sys
import numpy as np
import logging

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)

from RLutils.agent.Agent import Agent
from RLutils.algorithm.utils import softmax, softmax_power


class MDPrankAgent(Agent):

    def __init__(self, hyperparams, theta0):
        super(MDPrankAgent, self).__init__(hyperparams)
        self.theta = theta0
        self.nParam = len(theta0)
        return

    # perform an action according to the policy, given the current state
    # can be provided with a pre-calculated policy pi to improve the speed
    def act(self, state, random=True, pi=None):
        # if the policy is stochastic, this is a MonteCarlo sampling
        logging.debug("* acting: at t = %d, %d candidates" % (state[0], len(state[1])))
        actions = self.getActionList(state)

        if pi is None:
            pi = self.calPolicyProbMap(state)
        # assert np.abs(np.sum(pi) - 1.0) < 1.0e-5

        logging.debug("policy prob map: pi = [" + ','.join(["P(A" + str(ia) + ")=" + ("%.5f" % pa) for ia, pa in enumerate(pi)]) + "]")

        if random:
            randNum = np.random.rand(1)[0]  # a random number between [0, 1] according to the uniform distribution
            logging.debug("RANDOM act: generate random num %0.3f" % randNum)
            for action, action_prob in zip(actions, pi):
                randNum -= action_prob
                if randNum < 0:
                    logging.debug("choose A%d" % action)
                    return action
        else:
            ia = np.argmax(pi)
            logging.debug("Deterministic act: choose A%d with max prob = %0.3f" % (ia, np.max(pi)))
            return actions[ia]

        logging.warning("warning: have not selected an action; should NOT happen; simply choose A%d" % actions[-1])
        return actions[-1]


    # list of all possible actions given the current state
    def getActionList(self, state):
        candidates = state[1]
        return range(len(candidates))

    # the probability of action given a state
    def calPolicyProbMap(self, state):
        #assert self._hyperparams["policyTYpe"] == "stochastic"

        actions = self.getActionList(state)

        hvals = np.array([self.h(state, action) for action in actions])

        # pi(a|s), the probability of executed action given the current state
        if "softmax_power" not in self._hyperparams or self._hyperparams["softmax_power"] == 1:
            pi = softmax(hvals)
        else:
            power = int(self._hyperparams["softmax_power"])
            pi = softmax_power(hvals, power)

        self.pi = pi

        return pi

    # an evaluation function of state and action; current just use a linear model
    def h(self, state, action):
        candidates = state[1]

        # the list of candidates which are not ranked yet; each candidate contains its feature and label
        candidate = candidates[action]
        x = candidate[0]

        #assert self.nParam == x.shape[0]
        h = np.dot(self.theta, x)

        return h

    # fast solution for MDP rank; LOSS the generalization
    def cal_hvals_from_init(self, candidates):
        h_dict = {}
        for i, candidate in enumerate(candidates):
            x = candidate[0]
            h = np.dot(self.theta, x)
            h_dict.update({i: h})

        return h_dict

