#!/usr/bin/python
# -*- coding: <encoding name> -*-

import os
import sys
import copy

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)

from data.Letor.LectorDataDealer import getQueries, getQueryData
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

        t = state[0]  # step t, i.e. the t-th position in the rank case
        candidates = state[1]  # the list of candidates which are not ranked yet

        if self._hyperparams['reward_metric'] == 'NDCG':
            label = candidates[action][1]

            return DCG_singlePos(label, t)

        return 0

    # evolve to the next state
    def transit(self, state, action):
        t = state[0]
        candidates = state[1]

        assert 0 <= action < len(candidates)

        candidates_new = copy.deepcopy(candidates)
        del candidates_new[action]

        state_new = [t+1, candidates_new]

        return state_new

    def getQueries(self, mode="train"):
        if mode == "validation":
            return getQueries(self.validData)
        if mode == "test":
            return getQueries(self.testData)

        return getQueries(self.data)

    # dict of state keys and values
    def getCandidates(self, query, mode="train"):
        if mode == "validation":
            return getQueryData(self.validData, query)
        if mode == "test":
            return getQueryData(self.testData, query)

        return getQueryData(self.data, query)





if __name__ == "__main__":
    mdprankenvironment = MDPrankEnvironment({}, None)
