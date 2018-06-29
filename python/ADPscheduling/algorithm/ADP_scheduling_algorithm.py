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

from cvxpy import SolverError

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)

from RLutils.algorithm.ALGconfig import ALGconfig


class ADP_scheduling_algorithm(object):

    def __init__(self, hyperparams):
        config = copy.deepcopy(ALGconfig)
        config.update(hyperparams)
        self._hyperparams = config

        logging.info("%i iterations, learning rate = %f, discount rate = %f, CAVE step length = %f, slope adjust type = %s"
                     % (self._hyperparams['nIter'], self._hyperparams['eta'], self._hyperparams['discount'], self._hyperparams['cave_step'], self._hyperparams['cave_type']))
        if self._hyperparams['verbose']:
            print("%i iterations, learning rate = %f, discount rate = %f, CAVE step length = %f, slope adjust type = %s"
                  % (self._hyperparams['nIter'], self._hyperparams['eta'], self._hyperparams['discount'], self._hyperparams['cave_step'], self._hyperparams['cave_type']))

        self.agent = None
        self.env = None

        return

    def initAgent(self, agent):
        self.agent = agent
        logging.info("init agent")
        return

    def initEnv(self, env):
        self.env = env
        logging.info("init environment")
        return

    def offline_train(self, data):
        time_space_info, init_resource, task = data
        T = len(time_space_info["time"])
        n = len(time_space_info["location"])

        Vopt_results = np.zeros((self._hyperparams['nIter'], T))  # results of optimal values
        Vopt_results_noP = np.zeros((self._hyperparams['nIter'], T))  # results of optimal values
        Xopt_results = defaultdict(list)  # results of job decisions
        Yopt_results = defaultdict(list)  # results of relocation decisions
        CPUtime = np.zeros((self._hyperparams['nIter'], T))  # CPU time of each iteration, each simulation time
        Rout = defaultdict(list)  # results of resources at the end node
        pi_plus = defaultdict(list)  # the update right-side slope at all simulation times
        pi_minus = defaultdict(list)  # the update left-side slope at all simulation times

        # the multistage stochastic value approximation algorithm
        print('Simulation begins')
        state = init_resource

        for k in range(self._hyperparams['nIter']):
            print 'Iteration k = %i:' % k

            # Forward simulation
            for t in range(T):

                try:
                    startTime = datetime.datetime.now()

                except SolverError:
                    print 't = %i, Solver failed, use results in the last iteration!' % t
                else:
                    pass



        return

    def update_policy(self, delta_theta):
        logging.debug("Before update: theta = [" + ','.join([str(dt) for dt in self.agent.theta]) + ']')
        self.agent.theta = self.opt.Gradient_Descent(self.agent.theta, delta_theta, self._hyperparams["eta"])
        logging.debug("After update: theta = [" + ','.join([str(dt) for dt in self.agent.theta]) + ']')

        # if process new theta with sigmoid function, as indicated in the diversification paper
        if 'param_with_scale' in self._hyperparams:
            self.agent.theta = scaler(self.agent.theta, self._hyperparams['param_with_scale'])
            logging.debug("After scaling: theta = [" + ','.join([str(dt) for dt in self.agent.theta]) + ']')

        return


