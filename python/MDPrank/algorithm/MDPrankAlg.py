'''
The MDP algorithm
'''

import os
import sys
import copy
import numpy as np
import datetime

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)

from MDPrank.environment.MDPrankEnvironment import MDPrankEnvironment
from MDPrank.agent.MDPrankAgent import MDPrankAgent

from RLutils.algorithm.ALGconfig import ALGconfig
import RLutils.algorithm.policy_gradient
from RLutils.algorithm.Optimizer import Optimizer
from RLutils.algorithm.utils import softmax, sigmoid
from RLutils.environment.rankMetric import NDCG

class MDPrankAlg(object):

    def __init__(self, hyperparams):
        config = copy.deepcopy(ALGconfig)
        config.update(hyperparams)
        self._hyperparams = config

        if self._hyperparams['verbose']:
            print("learning rate = %f" % self._hyperparams['eta'])
            print("discount rate = %f" % self._hyperparams['discount'])

        self.agent = None
        self.env = None
        self.testenv = None
        self.opt = Optimizer(type='max')
        return

    def initAgent(self, agent):
        self.agent = agent
        return

    def initEnv(self, env):
        self.env = env
        return

    def learn(self, train_outputPath):

        time_start = datetime.datetime.now()

        # Algorithm 1 MDPrank learning
        queryList = self.env.getQueries()

        if self._hyperparams['verbose']:
            print("%d queries" % len(queryList))

        if 'nAbsErr' not in self._hyperparams:
            self._hyperparams['nAbsErr'] = 1
        iAbsErr = 0
        nStep = 0
        cpuTimes = []

        with open(train_outputPath, "w") as file:

            while(True):
                time_start_iter = datetime.datetime.now()

                delta_theta = np.zeros(self.agent.nParam)

                G0_allQueries = np.zeros(len(queryList))
                NDCG_allQueries = np.zeros(len(queryList))

                for iq, query in enumerate(queryList):

                    episode = self.sampleAnEpisode(query, offline=True)

                    M = len(episode)

                    labels = self.getLabelsFromEpisode(episode)
                    NDCG_allQueries[iq] = NDCG(labels)

                    for t in range(M):
                        Gt = self.calLongTermReturn(episode, t)

                        state, action, reward = episode[t]
                        grad_theta = self.calGradParam(state, action)
                        delta_theta = delta_theta + self._hyperparams["discount"]**t*Gt*grad_theta


                        if t == 0:
                            G0_allQueries[iq] = Gt

                        if self._hyperparams['verbose']:
                            if nStep == 0 and iq == 1 and t < 10 and False:
                                print("t = %d, action = %d, reward = %f, state1 = %d, state2_n = %d, Gt = %f" % (t, action, reward, state[0], len(state[1]), Gt))
                                print("grad of theta:")
                                print(grad_theta)
                                print('delta of theta:')
                                print(delta_theta)

                        #if t >= 10:
                            #break

                #self.agent.theta = self.agent.theta + self._hyperparams["eta"]*delta_theta
                #print("before:")
                #print(self.agent.theta)
                self.agent.theta = self.opt.Gradient_Descent(self.agent.theta, delta_theta, self._hyperparams["eta"])

                # if process new theta with sigmoid function, as indicated in the diversification paper
                if self._hyperparams['param_with_sigmoid']:
                    self.agent.theta = sigmoid(self.agent.theta)

                if self._hyperparams['verbose'] and False:
                    if nStep == 0 and iq == 1:
                        print("change:")
                        print(self._hyperparams["eta"]*delta_theta)
                        print('after')
                        print(self.agent.theta)

                cpuTime_iter = (datetime.datetime.now() - time_start_iter).total_seconds()

                # evaluate validation and test sets performance
                NDCG_mean_valid, NDCG_queries_valid = self.eval(dataSet="validation")
                NDCG_mean_test, NDCG_queries_test = self.eval(dataSet="test")

                cpuTimes.append(cpuTime_iter)

                # print iteration results
                outputData = [nStep, np.mean(G0_allQueries), np.mean(NDCG_allQueries), np.linalg.norm(delta_theta), NDCG_mean_valid, NDCG_mean_test, cpuTime_iter]
                outputData.extend(list(self.agent.theta))
                outputData = [str(d) for d in outputData]

                file.write('\t'.join(outputData) + '\n')
                if self._hyperparams['verbose']:
                    print("%dth iteration: averaged G0 = %0.3f, averaged NDCG = %0.3f, step norm = %0.3f, valid NDCG = %0.3f, test NDCG = %0.3f, compute time = %ds" % (nStep, np.mean(G0_allQueries), np.mean(NDCG_allQueries), np.linalg.norm(delta_theta),  NDCG_mean_valid, NDCG_mean_test, cpuTime_iter))

                if 'absErr' in self._hyperparams and np.linalg.norm(delta_theta) <= self._hyperparams['absErr']:
                    iAbsErr += 1

                    if iAbsErr >= self._hyperparams['nAbsErr']:
                        file.write("iterations terminate with absError reached\n")
                        if self._hyperparams['verbose']:
                            print("iterations terminate with absError reached")
                        break

                nStep += 1
                if 'iterations' in self._hyperparams and nStep > self._hyperparams['iterations']:
                    file.write("iterations terminate with max limit of iteration steps reached\n")

                    if self._hyperparams['verbose']:
                        print("iterations terminate with max limit of iteration steps reached")
                    break


            cpuTime_total = (datetime.datetime.now() - time_start).total_seconds()
            file.write("total %ds used\n" % cpuTime_total)
            print("total %0.2fs used" % cpuTime_total)

            file.close()

        return

    def eval(self, dataSet="test"):

        queryList = self.env.getQueries(dataSet)

        NDCG_queries = np.zeros(len(queryList))
        for iq, query in enumerate(queryList):
            episode = self.sampleAnEpisode(query, offline=False, dataSet=dataSet)
            labels = self.getLabelsFromEpisode(episode)
            NDCG_queries[iq] = NDCG(labels)
        NDCG_mean = np.mean(NDCG_queries)

        return NDCG_mean, NDCG_queries

    # the long-term return of the sampled episode starting from t, Equation (3)
    def calLongTermReturn(self, episode, t):
        Gt = 0.0

        discount_r = 1.0
        for k in range(t, len(episode)):
            rk = episode[k][2]
            Gt += discount_r*rk
            discount_r = discount_r*self._hyperparams["discount"]
        return Gt

    # the direction taht most increase the probability of repeating the action on future visits to state, Equation (4)
    def calGradParam(self, state, action):

        candidates = state[1]
        x = candidates[action][0]

        At = self.agent.getActionList(state)  # all possible actions with state t

        pi = self.agent.calPolicyProbMap(state)

        grad_theta = x
        for i, a in enumerate(At):
            grad_theta -= pi[i]*candidates[a][0]

        return grad_theta


    # Algorithm 2: SampleAnEpisode
    def sampleAnEpisode(self, query, offline=True, dataSet="train"):
        candidates = self.env.getCandidates(query, dataSet)

        state = [0, candidates]  # the initial state
        M = len(candidates)  # number of candidates to rank
        episode = list()  # the resulted episode: list of (s, a, r) for all time steps

        random = offline

        for t in range(M):
            #print("t = " + str(t))
            action = self.agent.act(state, random)  # Equation (2)

            reward = self.env.reward(state, action)  # Equation(1), calculated on the basis of label

            episode.append((state, action, reward))  # each instance is (s_t, a_t, r_{t+1})

            # continue to the next state
            state = self.env.transit(state, action)

        return episode

    def getLabelsFromEpisode(self, episode):

        labels = []
        for state, action, reward in episode:

            labels.append(state[1][action][1])
        return labels