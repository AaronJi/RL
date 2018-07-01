#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import logging

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)

from data.Time_Space import time_space_data_dealer
from RLutils.environment.Environment import Environment
from RLutils.environment.rankMetric import DCG_singlePos

class ADPschedulingEnvironment(Environment):

    def __init__(self, hyperparams, datadealer):
        super(ADPschedulingEnvironment, self).__init__(hyperparams)
        self.datadealer = datadealer
        return

    def set_environment_knowledge(self, time_space_info):
        self.time_space_info = time_space_info
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

    # yield reward given the current state and action
    def reward(self, state, action):
        return None

    # evolve to the next state
    def transit(self, state, action):
        return None

    # yield reward and evolve to the next state simultaneously
    def reward_and_transit(self, state, action, act_extra_factor):
        Rout, Vopt, pi_plus, pi_minus = act_extra_factor

        r = (Vopt, pi_plus, pi_minus)

        next_resource, next_incoming_resource = self.build_state_from_data(Rout)

        state_next = [state[0] + 1, next_resource, next_incoming_resource]

        return state_next, r

    # translate the data into the state format
    def build_state_from_data(self, Rout):
        # resource distrubution in the next step
        next_resource = {}
        next_incoming_resource = []

        n, max_period = Rout.shape
        for tau in range(max_period):
            next_incoming_resource.append([])

        for i in range(n):
            location_key = self.time_space_info['location_seq'][i]
            next_resource[location_key] = Rout[i, 0]

            for tau in range(max_period-1):
                if Rout[i, tau+1] > 0:
                    next_incoming_resource[tau].append({'destination': location_key, 'nR': Rout[i, tau+1]})

        return next_resource, next_incoming_resource


if __name__ == "__main__":
    ADPscheduling_env = ADPschedulingEnvironment({}, None)
