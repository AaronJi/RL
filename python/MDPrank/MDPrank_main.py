#!/usr/bin/python
# -*- coding: <encoding name> -*-

import logging
import imp
import os
import sys
import argparse
import threading
import datetime
import traceback
import numpy as np

# path of the whole project
MDPrank_main_path = os.path.abspath(__file__)
#project_dir = '/'.join(str.split(MDPrank_main_path, '/')[:-3]) + '/'
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(MDPrank_main_path))) + '/'
sys.path.append(project_dir)

from data.Letor.LectorDataDealer import LectorDataDealer
from algorithm.MDPrankAlg import MDPrankAlg
from agent.MDPrankAgent import MDPrankAgent
from environment.MDPrankEnvironment import MDPrankEnvironment


class MDPrankMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, args, init=None, sample=None):
        """
        Initialize GPSMain
        Args:
            config: Hyperparameters for experiment
            quit_on_end: When true, quit automatically on completion
        """

        if sample is None:
            self.sample = ""
        else:
            self.sample = '_' + str(sample)

        exp_name = args.experiment
        exp_dir = project_dir + 'experiments/' + exp_name + '/'
        hyperparams_file = exp_dir + 'hyperparams.py'

        data_dir = project_dir + 'data/'

        output_dir = project_dir + "/experiments/" + args.experiment + "/data_files/"
        self.train_outputPath = output_dir + args.train_output
        self.valid_outputPath = output_dir + args.valid_output
        self.test_outputPath = output_dir + args.test_output

        self.hyperparams = imp.load_source('hyperparams', hyperparams_file)
        if args.silent:
            self.hyperparams.config['verbose'] = False
            self.hyperparams.ALGconfig['verbose'] = False

        if args.silent:
            # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                                datefmt='%m-%d %H:%M',
                                filename=exp_dir + 'exp.log' + self.sample,
                                filemode='w')
        else:
            # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                                datefmt='%m-%d %H:%M',
                                filename=exp_dir + 'exp.log' + self.sample,
                                filemode='w')
        handler = logging.StreamHandler()
        LOGGER = logging.getLogger(__name__ + self.sample)
        LOGGER.addHandler(handler)
        LOGGER.fatal(datetime.datetime.now().strftime('%Y-%m-%d'))

        ## init algorithm module
        self.alg = MDPrankAlg(self.hyperparams.ALGconfig)

        ## init environment module
        self.lectordatadealer = LectorDataDealer(self.hyperparams.DATAconfig)
        trainingData_raw = self.lectordatadealer.load_data(data_dir + args.training_set)
        #trainingData = self.lectordatadealer.getPartData(trainingData_raw, 2, 2, 18)
        #trainingData = self.lectordatadealer.getPartData(trainingData_raw, 1, 2, 8)
        trainingData = trainingData_raw
        validationData = self.lectordatadealer.load_data(data_dir + args.valid_set)
        testData = self.lectordatadealer.load_data(data_dir + args.test_set)
        if self.hyperparams.config['verbose']:
            print("%d train data, %d validation data, %d test data" % (len(trainingData), len(validationData), len(testData)))
        self.nTheta = self.lectordatadealer.nFeature
        env = MDPrankEnvironment(self.hyperparams.ENVconfig, self.lectordatadealer)
        env.setTrainData(trainingData)
        env.setValidData(validationData)
        env.setTestData(testData)
        self.alg.initEnv(env)

        if self.hyperparams.config['verbose']:
            print("model has %d features" % self.nTheta)

        ## init agent module
        theta0 = np.zeros(self.nTheta)
        if init is None or init == "zero":
            pass
        elif init == "random":
            # the initial parameter of agent, with range [-1, 1]
            theta0 = np.random.rand(self.nTheta) * 2 - 1

            # write_theta = None
            write_theta = output_dir + "theta0.txt"
            if write_theta is not None:
                with open(write_theta, "w") as file:
                    for th in theta0:
                        file.write(str(th) + '\n')
                    file.close()
        else:
            # read theta from file
            try:
                with open(init, "r") as file:
                    theta0 = []
                    lines = file.readlines()

                    for i, line in enumerate(lines):
                        values = line.rstrip('\n').rstrip(']').lstrip('[').split(' ')
                        theta0.extend([float(v) for v in values if len(v) > 0])
                    file.close()

                    assert len(theta0) == self.nTheta
                    theta0 = np.array(theta0)
            except:
                pass
        if self.hyperparams.config['verbose']:
            print("init param:")
            print(theta0)
        agent = MDPrankAgent(self.hyperparams.AGEconfig, theta0)
        self.alg.initAgent(agent)

        LOGGER.info("init successfully")

        return

    def learn(self):
        train_outPath = self.train_outputPath + self.sample
        self.alg.learn(train_outPath)

        if self.hyperparams.config['verbose']:
            print("new param:")
            print(self.alg.agent.theta)

        return self.alg.agent.theta

    def eval(self, dataSet="test"):
        NDCG_mean, NDCG_queries = self.alg.eval(dataSet=dataSet)
        return NDCG_mean


def main():
    """ Main function to be run. """

    parser = argparse.ArgumentParser(description='Run the MDP rank algorithm.')

    parser.add_argument('experiment', type=str, help='experiment name')

    # TD2003/Fold1
    parser.add_argument('--training_set', type=str, default="Letor/OHSUMED/Data/Fold1/trainingset.txt",
                        help='training set')
    parser.add_argument('--valid_set', type=str, default="Letor/OHSUMED/Data/Fold1/validationset.txt",
                        help='validation set')
    parser.add_argument('--test_set', type=str, default="Letor/OHSUMED/Data/Fold1/testset.txt",
                        help='test set')

    parser.add_argument('--test_output', type=str, default="testoutput.txt",
                        help='test output')
    parser.add_argument('--train_output', type=str, default="trainoutput.txt",
                        help='train output')
    parser.add_argument('--valid_output', type=str, default="validoutput.txt",
                        help='valid output')

    parser.add_argument('--param_init', type=str, default="random",
                        help='valid output')  # "zero", "random",  output_dir + "theta1.txt"
    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')

    args = parser.parse_args()

    singleLearn(args, init_theta=args.param_init)
    #multiLearn(args, nLearner=10, init_theta=args.param_init)

    return

def singleLearn(args, init_theta="random"):

    # path to store experimental data
    output_dir = project_dir + "/experiments/" + args.experiment + "/data_files/"

    mdprank = MDPrankMain(args, init=init_theta)

    theta_new = mdprank.learn()
    output_path = output_dir + "theta1.txt"
    write_vector(theta_new, output_path)

    time0 = datetime.datetime.now()
    NDCG_mean = mdprank.eval(dataSet="validation")
    print("Evaluation: averaged NDCG of validation set = %0.3f" % NDCG_mean)
    print("%ds used" % ((datetime.datetime.now() - time0).total_seconds()))
    time0 = datetime.datetime.now()
    NDCG_mean = mdprank.eval(dataSet="test")
    print("Evaluation: averaged NDCG of test set = %0.3f" % NDCG_mean)
    print("%ds used" % ((datetime.datetime.now() - time0).total_seconds()))

    return

def multiLearn(args, nLearner=1, init_theta="random"):
    # path to store experimental data
    output_dir = project_dir + "/experiments/" + args.experiment + "/data_files/"

    learners = list()
    metric_valid = np.zeros(nLearner)
    metric_test = np.zeros(nLearner)
    for i in range(nLearner):
        learners.append(MDPrankMain(args, init=init_theta, sample=i))
        theta_new = learners[i].learn()
        output_path = output_dir + "theta1_" + str(i) + ".txt"
        write_vector(theta_new, output_path)

        metric_valid[i] = learners[i].eval(dataSet="validation")
        metric_test[i] = learners[i].eval(dataSet="test")

    print("Evaluation: averaged metric of validation set = %0.3f" % np.mean(metric_valid))
    print("Evaluation: averaged metric of test set = %0.3f" % np.mean(metric_test))
    write_vector(metric_valid, output_dir + "metric_valid.txt")
    write_vector(metric_test, output_dir + "metric_test.txt")
    return


def write_vector(vec, output_path):

    if output_path is not None:
        with open(output_path, "w") as file:
            for v in vec:
                file.write(str(v) + '\n')
            file.close()

    return

if __name__ == "__main__":
    main()
