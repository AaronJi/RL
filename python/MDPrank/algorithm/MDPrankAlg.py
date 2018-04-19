'''
The MDP algorithm
'''

import os
import sys
import copy
import numpy as np
import datetime
import logging

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)

from RLutils.algorithm.ALGconfig import ALGconfig
from RLutils.algorithm.policy_gradient import cal_policy_gradient, cal_longterm_ret
from RLutils.algorithm.Optimizer import Optimizer
from RLutils.algorithm.utils import softmax, softmax_power, scaler
from RLutils.environment.rankMetric import NDCG

class MDPrankAlg(object):

    def __init__(self, hyperparams):
        config = copy.deepcopy(ALGconfig)
        config.update(hyperparams)
        self._hyperparams = config

        logging.info("learning rate = %f, discount rate = %f" % (self._hyperparams['eta'], self._hyperparams['discount']))
        if self._hyperparams['verbose']:
            print("learning rate = %f, discount rate = %f" % (self._hyperparams['eta'], self._hyperparams['discount']))

        self.agent = None
        self.env = None
        self.testenv = None
        self.opt = Optimizer(type='max')
        return

    def initAgent(self, agent):
        self.agent = agent
        logging.info("init agent")
        return

    def initEnv(self, env):
        self.env = env
        logging.info("init environment")
        return

    def learn(self, train_outputPath):
        time_start = datetime.datetime.now()

        # Algorithm 1 MDPrank learning
        queryList = self.env.getQueries()

        logging.debug("#### Start learning, totally %d queries" % len(queryList))
        if self._hyperparams['verbose']:
            print("start learning, totally %d queries" % len(queryList))

        if 'nAbsErr' not in self._hyperparams:
            self._hyperparams['nAbsErr'] = 1

        iAbsErr = 0
        nStep = 0
        cpuTimes = []

        with open(train_outputPath, "w") as file:

            while(True):
                logging.debug("### The %dth iteration: ###" % nStep)
                time_start_iter = datetime.datetime.now()

                delta_theta = np.zeros(self.agent.nParam)

                G0_allQueries = np.zeros(len(queryList))
                NDCG_allQueries = np.zeros(len(queryList))

                for iq, query in enumerate(queryList):
                    logging.debug("## The %dth query: %s" % (iq, query))

                    if not self._hyperparams['fast_cal']:
                        episode = self.sampleAnEpisode(query, offline=True)
                    else:
                        episode, h_dict, grad_theta_list = self.sampleAnEpisode_fast(query, offline=True)

                    M = len(episode)

                    labels = self.getLabelsFromEpisode(episode)
                    logging.debug('episode label sequence: [' + ','.join([str(l) for l in labels]) + ']')

                    rewards = self.getRewardsFromEpisode(episode)
                    logging.debug('episode rewards sequence: [' + ','.join([str(r) for r in rewards]) + ']')

                    NDCG_allQueries[iq] = NDCG(labels)

                    for t in range(M-1):
                        # for the very last step, there is only one candidate to rank and only one possible action; the policy gradient then must be zero

                        Gt = cal_longterm_ret(rewards, t, self._hyperparams["discount"])

                        logging.debug("# step %d, Gt = %f" % (t, Gt))

                        if not self._hyperparams['fast_cal']:
                            state, action, reward = episode[t]
                            grad_theta = self.calGradParam(state, action)
                        else:
                            grad_theta = grad_theta_list[t]

                        logging.debug("Before: delta_theta = [" + ','.join([str(dt) for dt in delta_theta]) + ']')
                        delta_theta += cal_policy_gradient(t, Gt, self._hyperparams["discount"], grad_theta)
                        logging.debug("After: delta_theta = [" + ','.join([str(dt) for dt in delta_theta]) + ']')

                        if t == 0:
                            G0_allQueries[iq] = Gt

                logging.debug("Before update: theta = [" + ','.join([str(dt) for dt in self.agent.theta]) + ']')
                self.agent.theta = self.opt.Gradient_Descent(self.agent.theta, delta_theta, self._hyperparams["eta"])
                logging.debug("After update: theta = [" + ','.join([str(dt) for dt in self.agent.theta]) + ']')

                # if process new theta with sigmoid function, as indicated in the diversification paper
                if 'param_with_scale' in self._hyperparams:
                    self.agent.theta = scaler(self.agent.theta, self._hyperparams['param_with_scale'])

                    logging.debug("After scaling: theta = [" + ','.join([str(dt) for dt in self.agent.theta]) + ']')

                cpuTime_iter = (datetime.datetime.now() - time_start_iter).total_seconds()
                cpuTimes.append(cpuTime_iter)

                outputData = [nStep, cpuTime_iter, np.linalg.norm(delta_theta), np.mean(G0_allQueries), np.mean(NDCG_allQueries)]
                outputLine = "%dth iteration: compute time = %ds, step norm = %0.3f, averaged G0 = %0.3f, averaged metric = %0.3f" % (nStep, cpuTime_iter, np.linalg.norm(delta_theta), np.mean(G0_allQueries), np.mean(NDCG_allQueries))
                logging.debug("# after this iteration: compute time = %ds, norm of grad_theta = %0.3f, averaged G0 = %0.3f, averaged metric = %0.3f" % (cpuTime_iter, np.linalg.norm(delta_theta), np.mean(G0_allQueries), np.mean(NDCG_allQueries)))

                # evaluate validation and test sets performance
                if self._hyperparams['eval_valid_in_iters']:
                    NDCG_mean_valid, NDCG_queries_valid = self.eval(dataSet="validation")
                    outputData.append(NDCG_mean_valid)
                    outputLine += ", valid metric = %0.3f" % NDCG_mean_valid
                    logging.debug("Evaluate validation set: metric = %0.3f" % NDCG_mean_valid)
                if self._hyperparams['eval_test_in_iters']:
                    NDCG_mean_test, NDCG_queries_test = self.eval(dataSet="test")
                    outputData.append(NDCG_mean_test)
                    outputLine += ", test metric = %0.3f" % NDCG_mean_test
                    logging.debug("Evaluate test set: metric = %0.3f" % NDCG_mean_test)

                # print iteration results
                outputData.extend(list(self.agent.theta))
                outputData = [str(d) for d in outputData]
                file.write('\t'.join(outputData) + '\n')

                #if self._hyperparams['verbose']:
                print(outputLine)

                if 'absErr' in self._hyperparams and np.linalg.norm(delta_theta) <= self._hyperparams['absErr']:
                    iAbsErr += 1

                    if iAbsErr >= self._hyperparams['nAbsErr']:
                        file.write("iterations terminate with absError reached\n")
                        logging.debug("iterations terminate with absError reached")
                        if self._hyperparams['verbose']:
                            print("iterations terminate with absError reached")
                        break

                nStep += 1
                if 'iterations' in self._hyperparams and nStep > self._hyperparams['iterations']:
                    file.write("iterations terminate with max limit of iteration steps reached\n")
                    logging.debug("iterations terminate with max limit of iteration steps reached")
                    if self._hyperparams['verbose']:
                        print("iterations terminate with max limit of iteration steps reached")
                    break

            cpuTime_total = (datetime.datetime.now() - time_start).total_seconds()
            file.write("total %ds used\n" % cpuTime_total)
            logging.debug("learn is finished, total %ds used\n" % cpuTime_total)
            print("total %0.2fs used" % cpuTime_total)

            file.close()

        return

    def eval(self, dataSet="test"):
        queryList = self.env.getQueries(dataSet)
        logging.debug("#### start evaluation, totally %d queries" % len(queryList))

        NDCG_queries = np.zeros(len(queryList))
        for iq, query in enumerate(queryList):
            logging.debug("## The %dth query: %s" % (iq, query))
            if not self._hyperparams['fast_cal']:
                episode = self.sampleAnEpisode(query, offline=False, dataSet=dataSet)
            else:
                episode, h_dict, grad_theta_list = self.sampleAnEpisode_fast(query, offline=False, dataSet=dataSet)

            labels = self.getLabelsFromEpisode(episode)
            NDCG_queries[iq] = NDCG(labels)
        NDCG_mean = np.mean(NDCG_queries)

        return NDCG_mean, NDCG_queries

    # the long-term return of the sampled episode starting from t, Equation (3)
    def calLongTermReturn(self, episode, t):
        Gt = 0.0

        discount_rt = 1.0
        for k in range(t, len(episode)):
            rk = episode[k][2]
            Gt += discount_rt*rk
            discount_rt = discount_rt*self._hyperparams["discount"]
        return Gt

    # the direction that most increase the probability of repeating the action on future visits to state, Equation (4)
    def calGradParam(self, state, action):

        candidates = state[1]
        At = self.agent.getActionList(state)  # all possible actions with state t
        pi = self.agent.calPolicyProbMap(state)

        grad_theta = candidates[action][0]
        for i, a in enumerate(At):
            grad_theta -= pi[i]*candidates[a][0]

        logging.debug("# calculate gradient of param: [" + ','.join([str(gt) for gt in grad_theta]) + "]")

        return grad_theta

    # Algorithm 2: SampleAnEpisode
    def sampleAnEpisode(self, query, offline=True, dataSet="train"):
        logging.debug("*** sampling an episode")

        episode = list()  # the resulted episode: list of (s, a, r) for all time steps

        candidates = self.env.getCandidates(query, dataSet)

        state = [0, candidates]  # the initial state
        M = len(candidates)  # number of candidates to rank

        random = offline

        for t in range(M):
            logging.debug("** step %d:" % t)
            action = self.agent.act(state, random)  # Equation (2)

            reward = self.env.reward(state, action)  # Equation(1), calculated on the basis of label

            episode.append((state, action, reward))  # each instance is (s_t, a_t, r_{t+1})

            # continue to the next state
            state = self.env.transit(state, action)

        logging.debug('episode action sequence: [' + ','.join([str(action) for state, action, reward in episode]) + ']')

        return episode

    # faster speed; LOSS the generalization
    def sampleAnEpisode_fast(self, query, offline=True, dataSet="train"):
        logging.debug("*** sampling an episode")

        episode = list()  # the resulted episode: list of (s, a, r) for all time steps
        grad_theta_list = list()

        candidates = self.env.getCandidates(query, dataSet)
        M = len(candidates)  # number of candidates to rank

        curr_candidates_keyList = range(M)
        h_dict = self.agent.cal_hvals_from_init(candidates)

        state = [0, candidates]  # the initial state

        random = offline

        for t in range(M):
            logging.debug("** step %d:" % t)

            curr_h_list = [h_dict[k] for k in curr_candidates_keyList]

            if "softmax_power" not in self._hyperparams or self._hyperparams["softmax_power"] == 1:
                curr_pi = softmax(np.array(curr_h_list))
            else:
                power = int(self._hyperparams["softmax_power"])
                curr_pi = softmax_power(np.array(curr_h_list), power)

            action = self.agent.act(state, random, curr_pi)  # Equation (2)

            reward = self.env.reward(state, action)  # Equation(1), calculated on the basis of label

            episode.append((state, action, reward))  # each instance is (s_t, a_t, r_{t+1})

            At = self.agent.getActionList(state)
            grad_theta = state[1][action][0]
            for i, a in enumerate(At):
                grad_theta -= curr_pi[i] * state[1][a][0]
            grad_theta_list.append(grad_theta)

            # continue to the next state
            state = self.env.transit(state, action)
            del curr_candidates_keyList[action]

            #assert len(curr_candidates_keyList) == len(state[1])

        logging.debug('episode action sequence: [' + ','.join([str(action) for state, action, reward in episode]) + ']')

        return episode, h_dict, grad_theta_list


    def getLabelsFromEpisode(self, episode):

        labels = []
        for state, action, reward in episode:

            labels.append(state[1][action][1])
        return labels

    def getRewardsFromEpisode(self, episode):
        rewards = []
        for state, action, reward in episode:
            rewards.append(reward)
        return rewards


