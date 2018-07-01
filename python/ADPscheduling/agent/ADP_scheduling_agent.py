#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import logging
from collections import defaultdict
from cvxpy import SolverError

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)

from RLutils.agent.Agent import Agent
from RLutils.algorithm.scheduling_mp import scheduling_mp_sparse

class ADP_scheduling_agent(Agent):

    def __init__(self, hyperparams, T, n, nR, max_period):
        super(ADP_scheduling_agent, self).__init__(hyperparams)

        self.n = n
        self.T = T
        self.max_period = max_period

        # initialize the value function
        self.Qfun = {'PT': defaultdict(list),  # the slopes of value approximation functions at all simulation times; v[t](k) is v_t^k at time t, the kth interval
                     'PLenT': defaultdict(list),  # the interval length of value approximation functions at all simulation times; vLen[t](k) is u_t^k+1 - u_t^k, with slope of v[t](k)
                     'NT': defaultdict(list)}  # number of v elements for all iterations, simulation steps, and all locations
        # at the first step, initialize all v by 0
        for t in range(T):
            P = np.zeros((1, n * max_period))
            PLen = np.zeros((1, n * max_period))
            for tau in range(max_period):
                for i in range(n):
                    PLen[0, tau*n+i] = nR+1
            self.Qfun['PT'][t] = P
            self.Qfun['PLenT'][t] = PLen
            for tau in range(max_period):
                for i in range(n):
                    self.Qfun['NT'][(t, tau, i)] = 1

        self.decision_record = []
        for t in range(T):
            self.decision_record.append([])

        return

    def set_environment_knowledge(self, time_space_info, repositions):
        self.time_space_info = time_space_info
        self.possible_repositions = repositions
        return

    # TODO load the policy from a saved file
    def load_policy(self, policy_path):

        return


    # translate the data into the matrix format
    def build_matrix_data(self, state, tasks_t):
        current_resource = state[1]
        incoming_resource = state[2]

        # Rt
        Rt = np.zeros((self.n, 1))
        for i in range(self.n):
            location_key = self.time_space_info['location_seq'][i]
            Rt[i, 0] = current_resource[location_key]

        # Ru
        Ru = np.zeros((self.n, self.max_period))
        for tau in range(self.max_period):
            for incoming in incoming_resource[tau]:
                location_index = self.list_find(self.time_space_info['location_seq'], incoming['destination'])
                Ru[location_index, tau] += incoming['nR']

        # param_job
        param_job = np.zeros((5, len(tasks_t)))
        for j, task in enumerate(tasks_t):
            param_job[0, j] = self.list_find(self.time_space_info['location_seq'], task['start'])
            param_job[1, j] = self.list_find(self.time_space_info['location_seq'], task['destination'])
            param_job[2, j] += 1
            param_job[3, j] = task['income']
            param_job[4, j] = min(task['duration'], self.max_period)  # for task longer than the max period limit, manually cut it

        # param_rep
        param_rep = np.zeros((4, len(self.possible_repositions)))
        for j, reposition in enumerate(self.possible_repositions):
            param_rep[0, j] = self.list_find(self.time_space_info['location_seq'], reposition['start'])
            param_rep[1, j] = self.list_find(self.time_space_info['location_seq'], reposition['destination'])
            param_rep[2, j] = reposition['cost']
            param_rep[3, j] = reposition['duration']

        return Rt, Ru, param_job, param_rep

    def list_find(self, l, element):
        for i, e in enumerate(l):
            if e == element:
                return i
        return None

    # perform an action according to the policy, given the current state
    # can be provided with a pre-calculated policy pi to improve the speed
    def act(self, state, extra_factor):
        t = state[0]
        tasks_t = extra_factor

        Rt, Ru, param_job, param_rep = self.build_matrix_data(state, tasks_t)
        #print('***')
        #print('param_job')
        #print(param_job)
        #print('param_rep')
        #print(param_rep)

        try:
            Vopt, Xopt, Yopt, end_resource, lambda_right, lambda_left, status_right, status_left = \
                scheduling_mp_sparse(self.n, self.max_period, Rt, Ru, param_job, param_rep, self.Qfun['PT'][t], self.Qfun['PLenT'][t])

            logging.debug("* acting: at t = %d, status of solving the right problem: %s, status of solving the left problem: %s" % (t, status_right, status_left))
            if self._hyperparams['verbose']:
                print("* acting: at t = %d, status of solving the right problem: %s, status of solving the left problem: %s" % (t, status_right, status_left))

        except SolverError:
            logging.warning("* acting: at t = %d, solve error happens" % t)
            if self._hyperparams['verbose']:
                print("* acting: at t = %d, solve error happens" % t)

            # if solve failed, get result from the last iteration
            if t > 0:
                action = (self.decision_record[t][-1]['Xopt'], self.decision_record[t][-1]['Yopt'])
                act_extra_output = (self.decision_record[t][-1]['Xopt'], self.decision_record[t][-1]['Vopt'], self.decision_record[t][-1]['right lambda'], self.decision_record[t][-1]['left lambda'])
                self.decision_record[t].append(self.decision_record[t][-1])
            else:
                # if the first state, just do nothing; TODO set a trivial solution?
                Vopt = 0.0
                Xopt = np.zeros((self.n, self.n))
                Yopt = np.zeros((self.n, self.n))
                Rout = np.hstack((Rt, Ru[:, :-1]))
                lambda_right = np.zeros((self.n, self.max_period+1))
                lambda_left = np.zeros((self.n, self.max_period+1))

                action = (Xopt, Yopt)
                act_extra_output = (Rout, Vopt, lambda_right, lambda_left)
                self.decision_record[t].append({'Vopt': Vopt, 'Xopt': Xopt, 'Yopt': Yopt, 'end resource': Rout, 'right lambda': lambda_right, 'left lambda': lambda_left, 'right status': 'fail', 'left status': 'fail'})
        else:
            Rout = np.round(end_resource)

            action = (Xopt, Yopt)
            act_extra_output = (Rout, Vopt, lambda_right, lambda_left)

            self.decision_record[t].append({'Vopt': Vopt, 'Xopt': Xopt, 'Yopt': Yopt, 'end resource': Rout, 'right lambda': lambda_right, 'left lambda': lambda_left, 'right status': status_right, 'left status': status_left})
        return action, act_extra_output
