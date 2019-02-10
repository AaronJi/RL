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
from RLutils.algorithm.utils import list_find


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

        # initialize results
        train_results = dict()
        train_results['states'] = list()
        train_results['actions'] = list()
        train_results['rewards'] = list()
        train_results['out_resource'] = list()
        train_results['Vopt'] = np.zeros((self._hyperparams['nIter'], T))  # results of optimal values
        train_results['GMVopt'] = np.zeros((self._hyperparams['nIter'], T))  # results of optimal GMV
        train_results['Qfun'] = list()

        CPUtime_all = np.zeros((self._hyperparams['nIter'], T+1))  # CPU time of each iteration, each simulation time


        # list of max_period elements; the tau-th element contains all incoming resources which will arrive after tau steps
        incoming_resource = []
        for tau in range(self._hyperparams['max_period']):
            incoming_resource.append([])

        # the initial state
        init_state = [0, init_resource, incoming_resource]
        state = init_state

        ## the multistage stochastic value approximation algorithm
        print('Training begins')

        self.v0_t0_sum_iters = np.zeros(self._hyperparams['nIter'])

        ## iteration of DP
        for k in range(self._hyperparams['nIter']):
            if self._hyperparams['verbose']:
                print('Iteration k = %i' % k)

            states_iter = list()
            actions_iter = list()
            rewards_iter = list()
            out_resource_iter = list()

            ## Forward simulation
            for t in range(T):
                # get current data
                # time_t = tasks[t].keys()[0]
                tasks_t = tasks[t].values()[0]

                # at the first step, initialize the state
                if t == 0:
                    state = init_state

                startTime = datetime.datetime.now()

                # act
                action, act_extra_factor = self.agent.act(state, tasks_t)

                # transit & reward
                state_next, reward = self.environment.reward_and_transit(state, action, act_extra_factor)

                # result update
                states_iter.append(copy.deepcopy(state))
                actions_iter.append(copy.deepcopy(action))
                rewards_iter.append(copy.deepcopy(reward))
                out_resource_iter.append(copy.deepcopy(act_extra_factor[0]))
                train_results['Vopt'][k, t] = reward[0]
                train_results['GMVopt'][k, t] = reward[1]

                endTime = datetime.datetime.now()

                CPUtime = (endTime - startTime).total_seconds()
                CPUtime_all[k, t] = CPUtime

                state = state_next

            logging.warning("* action executed, cost %f seconds" % np.sum(CPUtime_all[k, :-1]))
            if self._hyperparams['verbose']:
                print("* action executed, cost %f seconds" % np.sum(CPUtime_all[k, :-1]))

            # iteration result update
            train_results['states'].append(states_iter)
            train_results['actions'].append(actions_iter)
            train_results['rewards'].append(rewards_iter)
            train_results['out_resource'].append(out_resource_iter)

            # policy update
            startTime = datetime.datetime.now()

            for t in range(T - 1):
                self.agent.policy_update(t, train_results['rewards'][-1], train_results['out_resource'][-1])

            Qfun_current = copy.deepcopy(self.agent.Qfun)
            train_results['Qfun'].append(Qfun_current)

            endTime = datetime.datetime.now()

            CPUtime = (endTime - startTime).total_seconds()
            CPUtime_all[k, -1] = CPUtime

            v_t0_sum = np.sum(self.agent.Qfun['vT'][0][0, 0:n])
            self.v0_t0_sum_iters[k] = v_t0_sum

            logging.warning("* policy updated, cost %f seconds" % CPUtime_all[k, -1])
            if self._hyperparams['verbose']:
                print("* policy updated, cost %f seconds" % CPUtime_all[k, -1])

        print('Training ends, total CPU time is %f seconds' % np.sum(CPUtime_all))

        return train_results


    def show_results(self, train_results, data):
        import matplotlib.pyplot as plt

        time_space_info, init_resource, tasks, relocations = data

        n = self.agent.n
        T = self.agent.T

        period = 1  # plot data at the 1st comming period

        Vopt = train_results['Vopt']
        GMVopt = train_results['GMVopt']
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

        #for tt in range(T):
            #vv = Qfun[-1]['vT'][tt][:, (period - 1) * n: period*n]
            #print('*** t = ' + str(tt))
            #print(vv)

            #for i in range(n):
            #    print('# location %i: ' % i + ','.join([str(vvv) for vvv in vv[:, i]]))

        ## plot of value function (the marginal value at resource = 0, the immediate coming period)
        reg_size = (4, 4)
        fontSize = 6
        markerSize = 3200
        alpha = 0.7
        vbounds = [0, 10]

        def plot_v_value(ax, Qfun, t_plot):
            v_value = np.zeros((n, 3))
            for loc_i in range(reg_size[0]):
                for loc_j in range(reg_size[1]):
                    i = loc_i * 4 + loc_j
                    v_value[i, 0] = loc_i - (reg_size[0] - 1) / 2.0
                    v_value[i, 1] = loc_j - (reg_size[1] - 1) / 2.0
                    v_value[i, 2] = Qfun[-1]['vT'][t_plot][0, (period - 1) * n + i]

            scatter = ax.scatter(v_value[:, 0], v_value[:, 1], c=v_value[:, 2], s=markerSize, marker='s',
                                  cmap=plt.cm.Oranges, vmin=vbounds[0], vmax=vbounds[1], alpha=alpha)
            plt.colorbar(scatter, orientation='vertical', shrink=1, pad=0.01)

            for i in range(n):
                ax.annotate(float('%0.1f' % v_value[i, 2]), (v_value[i, 0], v_value[i, 1]), size=fontSize)

            plt.title('V @ t=%i & nR=0 & tau=0' % t_plot)

            plt.xlim(-reg_size[0] / 2, reg_size[0] / 2)
            plt.ylim(-reg_size[1] / 2, reg_size[1] / 2)
            plt.xticks(range(-reg_size[0] / 2, reg_size[0] / 2 + 1))
            plt.yticks(range(-reg_size[1] / 2, reg_size[1] / 2 + 1))

            return

        #fig0 = plt.figure(0, figsize=(4.8, 4))
        fig0 = plt.figure(0, figsize=(11, 10))
        ax0 = fig0.add_subplot(221)
        plot_v_value(ax0, Qfun, t_plot=8)
        ax1 = fig0.add_subplot(222)
        plot_v_value(ax1, Qfun, t_plot=12)
        ax2 = fig0.add_subplot(223)
        plot_v_value(ax2, Qfun, t_plot=16)
        ax3 = fig0.add_subplot(224)
        plot_v_value(ax3, Qfun, t_plot=20)


        ## scheduling result

        def find_v(r, v, vlen):
            cumsum = 0
            vr = 0
            for k in range(vlen.shape[0]):
                cumsum += vlen[k]
                if np.round(r) < cumsum:
                    vr = v[k]
                    break
                elif np.round(r) == cumsum and k < vlen.shape[0] - 1:
                    vr = (v[k] + v[k + 1]) / 2.0
                    break
            return vr

        # plot of scheduling results
        def plot_scheduling_result(ax, Qfun, time_space_info, tasks, t_plot):
            state = train_results['states'][-1][t_plot]
            action = train_results['actions'][-1][t_plot]
            #out_resource = train_results['out_resource'][-1][t_plot]

            # num of resources
            nR_curr = np.zeros(n)
            for i in range(n):
                location_key = time_space_info['location_seq'][i]
                nR_curr[i] = state[1][location_key]

            # demand of tasks
            tasks_t = tasks[t_plot].values()[0]
            nTasks_curr = np.zeros(n)
            for task in tasks_t:
                i = list_find(time_space_info['location_seq'], task['start'])
                #j = list_find(time_space_info['location_seq'], task['destination'])
                #task_start = time_space_info['location_detail'][time_space_info['location_seq'][i]]
                #task_dest = time_space_info['location_detail'][time_space_info['location_seq'][j]]
                nTasks_curr[i] += 1

            related_resource_avail = nR_curr - nTasks_curr

            # plot of value function
            v_value = np.zeros((n, 3))

            for loc_i in range(reg_size[0]):
                for loc_j in range(reg_size[1]):
                    i = loc_i * 4 + loc_j
                    v_value[i, 0] = loc_i - (reg_size[0] - 1) / 2.0
                    v_value[i, 1] = loc_j - (reg_size[1] - 1) / 2.0

                    v = Qfun[-1]['vT'][t_plot][:, (period - 1) * n + i]
                    vLen = Qfun[-1]['vLenT'][t_plot][:, (period - 1) * n + i]
                    v_value[i, 2] = find_v(nR_curr[i], v, vLen)

            scatter = ax.scatter(v_value[:, 0], v_value[:, 1], c=v_value[:, 2], s=markerSize, marker='s',
                                  cmap=plt.cm.Oranges, vmin=vbounds[0], vmax=vbounds[1], alpha=alpha)

            plt.colorbar(scatter, orientation='vertical', shrink=1, pad=0.01)

            # plot of schedulings
            Yopt = action[1]
            schedulings = list()
            for i in range(n):
                for j in range(n):
                    if i != j and Yopt[i, j] > 0:
                        scheduling_start = time_space_info['location_detail'][time_space_info['location_seq'][i]]
                        scheduling_dest = time_space_info['location_detail'][time_space_info['location_seq'][j]]
                        scheduling_nR = Yopt[i, j]

                        drfit = - (reg_size[0] - 1) / 2.0
                        schedulings.append([scheduling_start[0]+drfit, scheduling_start[1]+drfit, scheduling_dest[0]+drfit, scheduling_dest[1]+drfit, scheduling_nR])

            if len(schedulings) > 0:
                schedulings = np.array(schedulings)
                schedulings[:, 2:4] = schedulings[:, 2:4] - schedulings[:, 0:2]
                quiver = ax.quiver(schedulings[:, 0], schedulings[:, 1], schedulings[:, 2], schedulings[:, 3] , schedulings[:, 4], angles='xy',
                                   scale_units='xy', scale=1, width=0.01, cmap=plt.cm.winter)

            for i in range(n):
                ax.annotate(float('%i' % int(related_resource_avail[i])), (v_value[i, 0], v_value[i, 1]), size=fontSize)

            plt.title('Scheduling with V @ nR & t=%i & tau=0' % t_plot)
            plt.xlim(-reg_size[0] / 2, reg_size[0] / 2)
            plt.ylim(-reg_size[1] / 2, reg_size[1] / 2)
            plt.xticks(range(-reg_size[0] / 2, reg_size[0] / 2 + 1))
            plt.yticks(range(-reg_size[1] / 2, reg_size[1] / 2 + 1))

            return

        fig0 = plt.figure(1, figsize=(11, 10))
        ax0 = fig0.add_subplot(221)
        plot_scheduling_result(ax0, Qfun, time_space_info, tasks, t_plot=8)
        ax1 = fig0.add_subplot(222)
        plot_scheduling_result(ax1, Qfun, time_space_info, tasks, t_plot=12)
        ax2 = fig0.add_subplot(223)
        plot_scheduling_result(ax2, Qfun, time_space_info, tasks, t_plot=16)
        ax3 = fig0.add_subplot(224)
        plot_scheduling_result(ax3, Qfun, time_space_info, tasks, t_plot=20)


        ## result at a specific location
        iters_plot = [0, 9, 19, 29]
        #iters_plot = [0, 39, 69, 99]
        #iters_plot = [0, 399, 699, 999]

        i_plot = 13
        t_plot = 20
        v_plot = Qfun[-1]['vT'][t_plot][:, (period-1)*n+i_plot]
        vLen_plot = Qfun[-1]['vLenT'][t_plot][:, (period-1)*n+i_plot]

        # plot points of v and vLen at location = i_plot and time = t_plot
        def plot_points_v(v_plot, vLen_plot):
            # find the num of plotted intervals where v is smaller than cutoff
            zero_cutoff = 0.01
            l_plot = v_plot.shape[0] - 1
            for l in range(v_plot.shape[0]):
                if v_plot[l] < zero_cutoff:
                    l_plot = l
                    break
            l_plot = min(l_plot, v_plot.shape[0] - 1)

            splot = v_plot[:l_plot + 1].copy()
            wplot = vLen_plot[:l_plot + 1].copy()

            if wplot[wplot.shape[0] - 1] > 20:
                wplot[wplot.shape[0] - 1] = wplot[wplot.shape[0] - 2] + 2

            awplot = np.zeros(l_plot + 2)
            mwplot = np.zeros(l_plot + 1)
            vplot = np.zeros(l_plot + 2)
            for k in range(1, l_plot + 2):
                awplot[k] = awplot[k - 1] + wplot[k - 1]
                vplot[k] = vplot[k - 1] + wplot[k - 1] * splot[k - 1]
            for k in range(l_plot + 1):
                mwplot[k] = (awplot[k] + awplot[k + 1]) / 2.0

            return mwplot, splot, awplot, vplot

        mwplot, splot, awplot, vplot = plot_points_v(v_plot, vLen_plot)

        v0_plot = np.zeros((self._hyperparams['nIter'], T))
        for k in range(self._hyperparams['nIter']):
            for t in range(T):
                v0_plot[k, t] = Qfun[k]['vT'][t][0, (period - 1) * n + i_plot]

        fig = plt.figure(2)

        ax1 = fig.add_subplot(311)
        ax1.plot(awplot, vplot, '-', mwplot, splot, 'o')
        plt.xlim(awplot[0], awplot[-1])
        plt.ylim(0, max(vplot) * 1.2)
        plt.xticks(awplot)
        plt.xlabel('num of resource')
        plt.ylabel('value')
        plt.legend(['value', 'marginal value'])

        ax2 = fig.add_subplot(312)
        ax2.plot(range(self._hyperparams['nIter']), v0_plot[:, 0], '-k',
                 range(self._hyperparams['nIter']), v0_plot[:, 6], '-g',
                 range(self._hyperparams['nIter']), v0_plot[:, 12], '-r',
                 range(self._hyperparams['nIter']), v0_plot[:, 18], '-b')
        plt.legend(['t=0', 't=6', 't=12', 't=18'])
        plt.xlabel('iterations')
        plt.ylabel('v0')

        ax3 = fig.add_subplot(313)
        ax3.plot(range(T), v0_plot[iters_plot[0]], '-k',
                 range(T), v0_plot[iters_plot[1]], '-g',
                 range(T), v0_plot[iters_plot[2]], '-r',
                 range(T), v0_plot[iters_plot[3]], '-b')
        plt.legend(['iter=%i' % iters_plot[0], 'iter=%i' % iters_plot[1], 'iter=%i' % iters_plot[2], 'iter=%i' % iters_plot[3]])
        plt.xlabel('time')
        plt.ylabel('v0')

        '''

        ## aggregated results
        plt.figure(3)
        plt.subplot(311)
        plt.plot(range(self._hyperparams['nIter'] + 1), v_t0_sum_iters[0], '-k',
                 range(self._hyperparams['nIter'] + 1), v_t0_sum_iters[6], '-g',
                 range(self._hyperparams['nIter'] + 1), v_t0_sum_iters[12], '-r',
                 range(self._hyperparams['nIter'] + 1), v_t0_sum_iters[18], '-b')
        plt.legend(['t=0', 't=6', 't=12', 't=18'])
        plt.title('sumed on all locations with coming_period=1')
        plt.xlabel('iterations')
        plt.ylabel('sum of v-slopes')
        plt.subplot(312)
        plt.plot(range(1, self._hyperparams['nIter'] + 1), Vopt[:, 0], '-k',
                 range(1, self._hyperparams['nIter'] + 1), Vopt[:, 6], '-g',
                 range(1, self._hyperparams['nIter'] + 1), Vopt[:, 12], '-r',
                 range(1, self._hyperparams['nIter'] + 1), Vopt[:, 18], '-b')
        plt.legend(['t=0', 't=6', 't=12', 't=18'])
        plt.xlabel('iterations')
        plt.ylabel('stepwise objective')
        plt.subplot(313)
        plt.plot(range(1, self._hyperparams['nIter'] + 1), GMVopt[:, 0], '-k',
                 range(1, self._hyperparams['nIter'] + 1), GMVopt[:, 6], '-g',
                 range(1, self._hyperparams['nIter'] + 1), GMVopt[:, 12], '-r',
                 range(1, self._hyperparams['nIter'] + 1), GMVopt[:, 18], '-b')
        plt.legend(['t=0', 't=6', 't=12', 't=18'])
        plt.xlabel('iterations')
        plt.ylabel('stepwise GMV')

        plt.figure(4)
        plt.subplot(311)
        plt.plot(range(T), v_sum_t_iters[iters_plot[0]], '-k',
                 range(T), v_sum_t_iters[iters_plot[1]], '-g',
                 range(T), v_sum_t_iters[iters_plot[2]], '-r',
                 range(T), v_sum_t_iters[iters_plot[3]], '-b')
        plt.legend(['iter=%i' % iters_plot[0], 'iter=%i' % iters_plot[1], 'iter=%i' % iters_plot[2],
                    'iter=%i' % iters_plot[3]])
        plt.title('sumed on all locations with coming_period=1')
        plt.xlabel('tie')
        plt.ylabel('sum of dvd0')
        plt.subplot(312)
        plt.plot(range(T), Vopt[iters_plot[0]], '-k',
                 range(T), Vopt[iters_plot[1]], '-g',
                 range(T), Vopt[iters_plot[2]], '-r',
                 range(T), Vopt[iters_plot[3]], '-b')
        plt.legend(['iter=%i' % iters_plot[0], 'iter=%i' % iters_plot[1], 'iter=%i' % iters_plot[2],
                    'iter=%i' % iters_plot[3]])
        plt.xlabel('time')
        plt.ylabel('stepwise objective')
        plt.subplot(313)
        plt.plot(range(T), GMVopt[iters_plot[0]], '-k',
                 range(T), GMVopt[iters_plot[1]], '-g',
                 range(T), GMVopt[iters_plot[2]], '-r',
                 range(T), GMVopt[iters_plot[3]], '-b')
        plt.legend(['iter=%i' % iters_plot[0], 'iter=%i' % iters_plot[1], 'iter=%i' % iters_plot[2],
                    'iter=%i' % iters_plot[3]])
        plt.xlabel('time')
        plt.ylabel('stepwise GMV')
        
        '''

        plt.show()

        return
