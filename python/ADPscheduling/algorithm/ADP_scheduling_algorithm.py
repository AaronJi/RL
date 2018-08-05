'''
The MDP algorithm
'''

import os
import sys
import copy
import numpy as np
import datetime
import logging
from collections import defaultdict
from multiprocessing import Pool

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)

from RLutils.algorithm.ALGconfig import ALGconfig


class ADP_scheduling_algorithm(object):

    def __init__(self, hyperparams):
        config = copy.deepcopy(ALGconfig)
        config.update(hyperparams)
        self._hyperparams = config

        logging.info("%i iterations, learning rate = %f, discount rate = %f" % (self._hyperparams['nIter'], self._hyperparams['eta'], self._hyperparams['discount']))
        if self._hyperparams['verbose']:
            print("%i iterations, learning rate = %f, discount rate = %f" % (self._hyperparams['nIter'], self._hyperparams['eta'], self._hyperparams['discount']))

        self.agent = None
        self.environment = None
        self.result_record = {'Qfun': []}

        return

    def initAgent(self, agent):
        self.agent = agent
        logging.info("init agent")
        return

    def initEnv(self, env):
        self.environment = env
        logging.info("init environment")
        return

    def max_period_check(self, tasks, max_period):
        for t in range(len(tasks)):
            time = tasks[t].values([1])
            tasks_t = tasks[t].values([0])
            for task in tasks_t:
                if task['duration'] > max_period:
                    tasks[t][task]['duration'] = max_period
        return tasks

    def offline_train(self, data):
        time_space_info, init_resource, tasks, relocations = data
        T = len(time_space_info["time_detail"])
        n = len(time_space_info["location_detail"])

        # maximum period check
        #tasks = self.max_period_check(tasks, self._hyperparams['max_period'])

        # initialize results
        train_results = dict()
        Vopt = np.zeros((self._hyperparams['nIter'], T))  # results of optimal values
        Qfun = list()

        Xopt_results = defaultdict(list)  # results of job decisions
        Yopt_results = defaultdict(list)  # results of relocation decisions
        Rout = defaultdict(list)  # results of resources at the end node
        pi_plus = defaultdict(list)  # the update right-side slope at all simulation times
        pi_minus = defaultdict(list)  # the update left-side slope at all simulation times


        CPUtime_all = np.zeros((self._hyperparams['nIter'], T+1))  # CPU time of each iteration, each simulation time

        ## at the first step, initialize the state

        # list of max_period elements; the tau-th element contains all incoming resources which will arrive after tau steps
        incoming_resource = []
        for tau in range(self._hyperparams['max_period']):
            incoming_resource.append([])

        state0 = [0, init_resource, incoming_resource]
        state = state0

        ## the multistage stochastic value approximation algorithm
        print('Training begins')


        self.v0_t0_sum_iters = np.zeros(self._hyperparams['nIter'])

        for k in range(self._hyperparams['nIter']):
            if self._hyperparams['verbose']:
                print('Iteration k = %i' % k)

            # Forward simulation
            for t in range(T):
                # get current data
                # time_t = tasks[t].keys()[0]
                tasks_t = tasks[t].values()[0]

                if t == 0:
                    state = state0

                startTime = datetime.datetime.now()

                action, act_extra_factor = self.agent.act(state, tasks_t)
                state_next, reward = self.environment.reward_and_transit(state, action, act_extra_factor)

                endTime = datetime.datetime.now()

                CPUtime = (endTime - startTime).total_seconds()
                CPUtime_all[k, t] = CPUtime

                Vopt[k, t] = reward[0]

                state = state_next

            logging.warning("* action executed, cost %f seconds" % np.sum(CPUtime_all[k, :-1]))
            if self._hyperparams['verbose']:
                print("* action executed, cost %f seconds" % np.sum(CPUtime_all[k, :-1]))

            # policy update
            startTime = datetime.datetime.now()

            self.agent.policy_update()

            Qfun_current = copy.deepcopy(self.agent.Qfun)
            self.result_record['Qfun'].append(Qfun_current)
            Qfun.append(Qfun_current)

            endTime = datetime.datetime.now()

            CPUtime = (endTime - startTime).total_seconds()
            CPUtime_all[k, -1] = CPUtime

            v_t0_sum = np.sum(self.agent.Qfun['vT'][0][0, 0:n])
            self.v0_t0_sum_iters[k] = v_t0_sum

            logging.warning("* policy updated, cost %f seconds" % CPUtime_all[k, -1])
            if self._hyperparams['verbose']:
                print("* policy updated, cost %f seconds" % CPUtime_all[k, -1])

        print('Training ends, total CPU time is %f seconds' % np.sum(CPUtime_all))

        train_results['Vopt'] = Vopt
        train_results['Qfun'] = Qfun
        return train_results


    def show_results(self, train_results):
        import matplotlib.pyplot as plt

        n = self.agent.n
        T = self.agent.T
        period = 1

        Vopt = train_results['Vopt']
        Qfun = train_results['Qfun']

        v_sum_t_iters = []
        for k in range(self._hyperparams['nIter']):

            v_sum_t = np.zeros(T)
            for t in range(T):
                v_sum_t[t] = np.sum(Qfun[k]['vT'][t][0, (period - 1) * n:period * n])
                np.sum(self.agent.Qfun['vT'][0][0, 0:n])

            v_sum_t_iters.append(v_sum_t)

        v_t0_sum_iters = []
        for t in range(T):
            v_t0_sum = np.zeros(self._hyperparams['nIter'] + 1)
            for k in range(self._hyperparams['nIter']):
                v_t0_sum[k + 1] = np.sum(Qfun[k]['vT'][t][0, (period - 1) * n:period * n])
            v_t0_sum_iters.append(v_t0_sum)

        # result at a specific location
        plt.figure(0)
        i_plot = 1
        v0_plot = np.zeros((self._hyperparams['nIter'], T))
        for k in range(self._hyperparams['nIter']):
            for t in range(T):
                v0_plot[k, t] = Qfun[k]['vT'][t][0, (period - 1) * n + i_plot]
        plt.figure(0)
        plt.subplot(211)
        plt.plot(range(self._hyperparams['nIter']), v0_plot[:, 0], '-k',
                 range(self._hyperparams['nIter']), v0_plot[:, 6], '-g',
                 range(self._hyperparams['nIter']), v0_plot[:, 12], '-r',
                 range(self._hyperparams['nIter']), v0_plot[:, 18], '-b')
        plt.legend(['t=0', 't=6', 't=12', 'it=18'])
        plt.xlabel('iterations')
        plt.ylabel('v0')
        plt.subplot(212)
        plt.plot(range(T), v0_plot[0], '-k',
                 range(T), v0_plot[9], '-g',
                 range(T), v0_plot[19], '-r',
                 range(T), v0_plot[29], '-b')
        plt.legend(['iter 1', 'iter 10', 'iter 20', 'iter 30'])
        plt.xlabel('time')
        plt.ylabel('v0')

        plt.figure(1)
        plt.subplot(211)
        plt.plot(range(self._hyperparams['nIter'] + 1), v_t0_sum_iters[0], '-k',
                 range(self._hyperparams['nIter'] + 1), v_t0_sum_iters[6], '-g',
                 range(self._hyperparams['nIter'] + 1), v_t0_sum_iters[12], '-r',
                 range(self._hyperparams['nIter'] + 1), v_t0_sum_iters[18], '-b')
        plt.legend(['t=0', 't=6', 't=12', 'it=18'])
        plt.title('Sum of value function slopes on all locations (coming_period=1) as function of iteration')
        plt.xlabel('iterations')
        plt.ylabel('sum of dvd0')
        plt.subplot(212)
        plt.plot(range(1, self._hyperparams['nIter'] + 1), Vopt[:, 0], '-k',
                 range(1, self._hyperparams['nIter'] + 1), Vopt[:, 6], '-g',
                 range(1, self._hyperparams['nIter'] + 1), Vopt[:, 12], '-r',
                 range(1, self._hyperparams['nIter'] + 1), Vopt[:, 18], '-b')
        plt.legend(['t=0', 't=6', 't=12', 'it=18'])
        plt.title('Optimized object values as function of iteration')
        plt.xlabel('iterations')
        plt.ylabel('optimal object value')

        plt.figure(2)
        plt.subplot(211)
        plt.plot(range(T), v_sum_t_iters[0], '-k',
                 range(T), v_sum_t_iters[9], '-g',
                 range(T), v_sum_t_iters[19], '-r',
                 range(T), v_sum_t_iters[29], '-b')
        plt.legend(['iter 1', 'iter 10', 'iter 20', 'iter 30'])
        plt.title('Sum of value function slopes on all locations (coming_period=1) as function of t')
        plt.xlabel('time')
        plt.ylabel('sum of dvd0')
        plt.subplot(212)
        plt.plot(range(T), Vopt[0], '-k',
                 range(T), Vopt[9], '-g',
                 range(T), Vopt[19], '-r',
                 range(T), Vopt[29], '-b')
        plt.legend(['iter 1', 'iter 10', 'iter 20', 'iter 30'])
        plt.title('Optimized object values as function of t')
        plt.xlabel('time')
        plt.ylabel('optimal object value')

        #plt.figure(2)
        #plt.subplots(4, 4, 0)


        plt.show()

        return
