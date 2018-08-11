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
from RLutils.algorithm.cave import CAVE, v2A, A2v

class ADP_scheduling_agent(Agent):

    def __init__(self, hyperparams, T, n, nR, max_period):
        super(ADP_scheduling_agent, self).__init__(hyperparams)

        self.n = n
        self.T = T
        self.max_period = max_period

        # initialize the value function
        self.Qfun = {'vT': defaultdict(list),  # the slopes of value approximation functions at all simulation times; v[t](k) is v_t^k at time t, the kth interval
                     'vLenT': defaultdict(list),  # the interval length of value approximation functions at all simulation times; vLen[t](k) is u_t^k+1 - u_t^k, with slope of v[t](k)
                     'NT': defaultdict(list)}  # number of v elements for all iterations, simulation steps, and all locations
        # at the first step, initialize all v by 0
        for t in range(T):
            v = np.zeros((1, n * max_period))
            vLen = np.zeros((1, n * max_period))
            for tau in range(max_period):
                for i in range(n):
                    vLen[0, tau*n+i] = nR+1
            self.Qfun['vT'][t] = v
            self.Qfun['vLenT'][t] = vLen
            for tau in range(max_period):
                for i in range(n):
                    self.Qfun['NT'][(t, tau, i)] = 1

        # record of past planning decisions
        self.decision_record = []
        for t in range(T):
            self.decision_record.append([])

        logging.info(" CAVE step length = %f, slope adjust type = %s" % (self._hyperparams['cave_step'], self._hyperparams['cave_type']))
        #if self._hyperparams['verbose']:
        #    print("CAVE step length = %f, slope adjust type = %s" % (self._hyperparams['cave_step'], self._hyperparams['cave_type']))

        return

    def set_environment_knowledge(self, time_space_info, repositions):
        self.time_space_info = time_space_info
        self.possible_repositions = repositions
        return

    # TODO: load the policy from a saved file
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
        try:
            Vopt, GMVopt, Xopt, Yopt, end_resource, lambda_right, lambda_left, status_right, status_left = \
                scheduling_mp_sparse(self.n, self.max_period, Rt, Ru, param_job, param_rep, self.Qfun['vT'][t], self.Qfun['vLenT'][t], self._hyperparams['solver'])

            logging.debug("* acting: at t = %d, status of solving the right problem: %s, status of solving the left problem: %s" % (t, status_right, status_left))
            #if self._hyperparams['verbose']:
                #print("* acting: at t = %d, status of solving the right problem: %s, status of solving the left problem: %s" % (t, status_right, status_left))

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
                GMVopt = 0.0
                Xopt = np.zeros((self.n, self.n))
                Yopt = np.zeros((self.n, self.n))
                Rout = np.hstack((Rt, Ru[:, :-1]))
                lambda_right = np.zeros((self.n, self.max_period+1))
                lambda_left = np.zeros((self.n, self.max_period+1))

                action = (Xopt, Yopt)
                act_extra_output = (Rout, Vopt, lambda_right, lambda_left)
                self.decision_record[t].append({'Vopt': Vopt, 'GMVopt': GMVopt, 'Xopt': Xopt, 'Yopt': Yopt, 'Rout': Rout, 'right lambda': lambda_right, 'left lambda': lambda_left, 'right status': 'fail', 'left status': 'fail'})
        else:
            Rout = np.round(end_resource)

            action = (Xopt, Yopt)
            act_extra_output = (Rout, Vopt, lambda_right, lambda_left)

            self.decision_record[t].append({'Vopt': Vopt, 'GMVopt': GMVopt, 'Xopt': Xopt, 'Yopt': Yopt, 'Rout': Rout, 'right lambda': lambda_right, 'left lambda': lambda_left, 'right status': status_right, 'left status': status_left})
        return action, act_extra_output

    def policy_update(self):

        # determination of breakpoints
        for t in range(self.T - 1):
            logging.warning("* updating value function: determine new breakpoints at step = %i" % t)
            #if self._hyperparams['verbose']:
                #print("* updating value function: determine new breakpoints, at step = %i" % t)
            t_pi_o_minus, t_pi_o_plus = self.__pi_update(t)

            logging.warning("* updating value function: produce new slopes at step = %i" % t)
            #if self._hyperparams['verbose']:
                #print("* updating value function: produce new slopes, at step = %i" % t)
            self.__cave_update(t_pi_o_minus, t_pi_o_plus, t)

        return

    def __pi_update(self, t):

        # determination of the updating right-derivative and left-derivative
        t_pi_o_plus = np.zeros((self.n, self.max_period))  # each column means tau = 1, 2, ..., tau_max
        t_pi_o_minus = np.zeros((self.n, self.max_period))  # each column means tau = 1, 2, ..., tau_max

        if self._hyperparams['cave_type'] == 'DUALNEXT':
            # DUALNEXT: the slope update method of eq(17-18), Godfrey, Powell, 2002
            next_pi_plus = self.decision_record[t+1][-1]['right lambda']
            next_pi_minus = self.decision_record[t+1][-1]['left lambda']

            for tau in range(self.max_period):
                for i in range(self.n):
                    t_pi_o_plus[i][tau] = next_pi_plus[i][tau]
                    t_pi_o_minus[i][tau] = next_pi_minus[i][tau]
        elif self._hyperparams['cave_type'] == 'DUALMAX':
            # DUALMAX: the slope update method of eq(15-16), Godfrey, Powell, 2002
            future_pi_plus = self.decision_record[t+1][-1]['right lambda']
            future_pi_minus = self.decision_record[t+1][-1]['left lambda']
            for tau in range(self.max_period):
                candi_pi_plus = future_pi_plus[:, tau].reshape((self.n, 1))
                candi_pi_minus = future_pi_minus[:, tau].reshape((self.n, 1))
                if tau > 0:
                    # then candi_pi has more than one column
                    for s in range(1, tau + 1):
                        if t + 1 + s >= self.T:
                            break
                        future_pi_plus = self.decision_record[t+1+s][-1]['right lambda']
                        candi_pi_plus = np.hstack((candi_pi_plus, future_pi_plus[:, tau - s].reshape((self.n, 1))))
                        future_pi_minus = self.decision_record[t+1+s][-1]['left lambda']
                        candi_pi_minus = np.hstack((candi_pi_minus, future_pi_minus[:, tau - s].reshape((self.n, 1))))
                for i in range(self.n):
                    t_pi_o_plus[i][tau] = np.max(candi_pi_plus[i, :])
                    t_pi_o_minus[i][tau] = np.min(candi_pi_minus[i, :])
        else:
            print >> sys.stderr, 'Wrong definition of adjustment method!'
        return t_pi_o_minus, t_pi_o_plus

    ## Value function update with CAVE
    def __cave_update(self, t_pi_o_minus, t_pi_o_plus, t):

        # update v and vLen at each t using CAVE algorithm
        v = self.Qfun['vT'][t]  # self.vT[t]
        vLen = self.Qfun['vLenT'][t]  # self.vLenT[t]

        for tau in range(self.max_period):
            for i in range(self.n):
                icol = tau * self.n + i

                newBreakPoint = [self.decision_record[t][-1]['Rout'][i][tau], t_pi_o_minus[i][tau], t_pi_o_plus[i][tau]]

                A = v2A(v[:, icol], vLen[:, icol], self.Qfun['NT'][t, tau, i])  # convert to A from v and vLen
                A = CAVE(A, newBreakPoint, self._hyperparams['cave_step'])  # update A by CAVE
                v_vec, vLen_vec, N_vec = A2v(A)  # convert to v and vLen at each i and tau from A

                # update v and vLen; need to consider the consistence of matrix size
                N = v.shape[0]
                if N_vec <= N:
                    for k in range(N_vec, N):
                        v[k][icol] = 0
                        vLen[k][icol] = 0
                else:
                    v = np.vstack((v, np.zeros((N_vec - N, v.shape[1]))))
                    vLen = np.vstack((vLen, np.zeros((N_vec - N, v.shape[1]))))

                for k in range(N_vec):
                    v[k][icol] = v_vec[k]
                    vLen[k][icol] = vLen_vec[k]

                # store the new N
                self.Qfun['NT'][(t, tau, i)] = N_vec

        assert v.shape == vLen.shape

        # store updated v and vLen
        Ndiff = v.shape[0] - self.Qfun['vT'][t].shape[0]
        if Ndiff > 0:
            self.Qfun['vT'][t] = np.vstack((self.Qfun['vT'][t], np.zeros((Ndiff, self.max_period * self.n))))
            self.Qfun['vLenT'][t] = np.vstack((self.Qfun['vLenT'][t], np.zeros((Ndiff, self.max_period * self.n))))
        for tau in range(self.max_period):
            for i in range(self.n):
                icol = tau * self.n + i
                for N in range(v.shape[0]):
                    self.Qfun['vT'][t][N, icol] = v[N, icol]
                    self.Qfun['vLenT'][t][N, icol] = vLen[N, icol]

        return
