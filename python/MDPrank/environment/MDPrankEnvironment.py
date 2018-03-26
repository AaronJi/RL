#!/usr/bin/python
# -*- coding: <encoding name> -*-

import os
import sys
import copy
import logging

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)

from data.rec_data_dealer import RecDataDealer
from RLutils.environment.rankEnvironment import RankEnvironment
from RLutils.environment.rankMetric import DCG_singlePos


class MDPrankEnvironment(RankEnvironment):

    def __init__(self, hyperparams, datadealer):
        super(MDPrankEnvironment, self).__init__(hyperparams)
        self.datadealer = datadealer
        self.query = None
        return

    # set the current data set, i.e., training, validation, test data sets
    def setTrainData(self, data):
        self.data = data
        return

    def setValidData(self, data):
        self.validData = data
        return

    def setTestData(self, data):
        self.testData = data
        return

    # action: the index of candidate to rank next
    def reward(self, state, action):
        r = 0
        t = state[0]  # step t, i.e. the t-th position in the rank case
        candidates = state[1]  # the list of candidates which are not ranked yet
        label = candidates[action][1]
        if self._hyperparams['reward_metric'] == 'NDCG':
            r = DCG_singlePos(label, t)

        logging.debug("* rewarding: %dth step, %d candidates in state; action is %d; label = %f, reward = %f" % (t, len(candidates), action, label, r))
        return r

    # evolve to the next state
    def transit(self, state, action):
        t = state[0]
        candidates = state[1]

        #assert 0 <= action < len(candidates)

        candidates_new = copy.deepcopy(candidates)
        del candidates_new[action]
        logging.debug("* transiting: %d to %d, %d candidates to %d candidates" % (t, t+1, len(candidates), len(candidates_new)))
        state_new = [t+1, candidates_new]

        return state_new

    def getQueries(self, mode="train"):
        if mode == "validation":
            return RecDataDealer.getQueries(self.validData)
        if mode == "test":
            return RecDataDealer.getQueries(self.testData)

        return RecDataDealer.getQueries(self.data)

    # dict of state keys and values
    def getCandidates(self, query, mode="train"):
        if mode == "validation":
            return RecDataDealer.getQueryData(self.validData, query)
        if mode == "test":
            return RecDataDealer.getQueryData(self.testData, query)

        return RecDataDealer.getQueryData(self.data, query)





if __name__ == "__main__":
    mdprankenvironment = MDPrankEnvironment({}, None)
