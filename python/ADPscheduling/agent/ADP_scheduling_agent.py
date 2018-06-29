#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import logging
from collections import defaultdict

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)

from RLutils.agent.Agent import Agent
from RLutils.algorithm.scheduling_mp import scheduling_mp

class ADP_scheduling_agent(Agent):

    def __init__(self, hyperparams, T, n, nR):
        super(ADP_scheduling_agent, self).__init__(hyperparams)

        # initialize the value function
        self.Qfun = {'vT': defaultdict(list),  # the slopes of value approximation functions at all simulation times; v[t](k) is v_t^k at time t, the kth interval
                     'vLenT': defaultdict(list),  # the interval length of value approximation functions at all simulation times; vLen[t](k) is u_t^k+1 - u_t^k, with slope of v[t](k)
                     'NT': defaultdict(list)}  # number of v elements for all iterations, simulation steps, and all locations
        # at the first step, initialize all v by 0
        for t in range(T):
            v = np.zeros((1, n * self._hyperparams['max_period']))
            vLen = np.zeros((1, n * self._hyperparams['max_period']))
            for tau in range(self._hyperparams['max_period']):
                for i in range(n):
                    vLen[0, tau*n+i] = nR+1
            self.Qfun['vT'][t] = v
            self.Qfun['vLenT'][t] = vLen
            for tau in range(self._hyperparams['max_period']):
                for i in range(n):
                    self.Qfun['NT'][(t, tau, i)] = 1

        return

    # TODO load the policy from a saved file
    def load_policy(self, policy_path):

        return


    # perform an action according to the policy, given the current state
    # can be provided with a pre-calculated policy pi to improve the speed
    def act(self, state, extra_factor):
        Vopt, Xopt, Yopt, end_resource, lambda_right, lambda_left, status_right, status_left = \
            scheduling_mp(n, self._hyperparams['max_period'], Rt, Ru, param_job, param_rep, v, vLen)

        # if the policy is stochastic, this is a MonteCarlo sampling
        logging.debug("* acting: at t = %d, %d candidates" % (state[0], len(state[1])))
        actions = self.getActionList(state)

        if pi is None:
            pi = self.calPolicyProbMap(state)
        # assert np.abs(np.sum(pi) - 1.0) < 1.0e-5

        logging.debug("policy prob map: pi = [" + ','.join(["P(A" + str(ia) + ")=" + ("%.5f" % pa) for ia, pa in enumerate(pi)]) + "]")

        if self._hyperparams["policyType"] == "deterministic":
            random = False

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

        #assert nLocationParam == x.shape[0]
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

    # predit a single score
    def score_pointwise(self, x):
        return np.exp(np.dot(self.theta, x))

