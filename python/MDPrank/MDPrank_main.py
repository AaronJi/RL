#!/usr/bin/python
# -*- coding: <encoding name> -*-

import imp
import os
import sys
import argparse
import threading
import datetime
import traceback
import numpy as np
import logging

# path of the whole project
MDPrank_main_path = os.path.abspath(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(MDPrank_main_path)))
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
        exp_dir = os.path.join(project_dir, 'experiments', exp_name)
        hyperparams_file = os.path.join(exp_dir, 'hyperparams.py')

        if args.silent:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s',
                                datefmt='%m-%d %H:%M',
                                filename=os.path.join(exp_dir, "exp" + self.sample + ".log"),
                                filemode='w')
        else:
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s',
                                datefmt='%m-%d %H:%M',
                                filename=os.path.join(exp_dir, "exp" + self.sample + ".log"),
                                filemode='w')

        data_dir = os.path.join(project_dir, 'data')

        output_dir = os.path.join(project_dir, "experiments", args.experiment, "data_files")
        self.train_outputPath = os.path.join(output_dir, args.train_output)
        self.valid_outputPath = os.path.join(output_dir, args.valid_output)
        self.test_outputPath = os.path.join(output_dir, args.test_output)

        self.hyperparams = imp.load_source('hyperparams', hyperparams_file)
        if args.silent:
            self.hyperparams.config['verbose'] = False
            self.hyperparams.ALGconfig['verbose'] = False

        ## init algorithm module
        self.alg = MDPrankAlg(self.hyperparams.ALGconfig)

        ## init environment module
        self.lectordatadealer = LectorDataDealer(self.hyperparams.DATAconfig)
        trainingData_raw = self.lectordatadealer.load_data(os.path.join(data_dir, args.training_set))
        #trainingData = self.lectordatadealer.getPartData(trainingData_raw, 2, 2, 18)
        #trainingData = self.lectordatadealer.getPartData(trainingData_raw, 1, 2, 8)
        trainingData = trainingData_raw
        validationData = self.lectordatadealer.load_data(os.path.join(data_dir, args.valid_set))
        testData = self.lectordatadealer.load_data(os.path.join(data_dir, args.test_set))

        logging.info("%d train data, %d validation data, %d test data" % (len(trainingData), len(validationData), len(testData)))
        if self.hyperparams.config['verbose']:
            print("%d train data, %d validation data, %d test data" % (len(trainingData), len(validationData), len(testData)))


        self.nTheta = self.lectordatadealer.nFeature
        env = MDPrankEnvironment(self.hyperparams.ENVconfig, self.lectordatadealer)
        env.setTrainData(trainingData)
        env.setValidData(validationData)
        env.setTestData(testData)
        self.alg.initEnv(env)

        logging.info("model has %d features" % self.nTheta)
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
            write_theta = os.path.join(output_dir, "theta0.txt")
            if write_theta is not None:
                with open(write_theta, "w") as file:
                    for th in theta0:
                        file.write(str(th) + '\n')
                    file.close()
        else:
            # read theta from file
            try:
                read_theta = os.path.join(output_dir, init)

                with open(read_theta, "r") as file:
                    theta0 = []
                    lines = file.readlines()

                    for i, line in enumerate(lines):
                        values = line.rstrip('\n').rstrip(']').lstrip('[').split(' ')
                        theta0.extend([float(v) for v in values if len(v) > 0])
                    file.close()

                    # assert len(theta0) == self.nTheta
                    theta0 = np.array(theta0[:self.nTheta])
            except:
                pass

        logging.info("init param: " + ','.join([str(th) for th in theta0]))
        if self.hyperparams.config['verbose']:
            print("init param:")
            print(theta0)

        agent = MDPrankAgent(self.hyperparams.AGEconfig, theta0)
        self.alg.initAgent(agent)

        logging.info("Init successfully")

        return

    def learn(self):
        train_outPath = self.train_outputPath + self.sample
        self.alg.learn(train_outPath)

        logging.info("new param after learning: " + ','.join([str(th) for th in self.alg.agent.theta]))
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
                        help='param init')  # "zero", "random",  output_dir + "theta1.txt"
    parser.add_argument('--param_out', type=str, default="theta1.txt",
                        help='param out')

    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')

    args = parser.parse_args()


    singleLearn(args, init_theta=args.param_init, out_theta=args.param_out)
    #multiLearn(args, nLearner=10, init_theta=args.param_init)

    return

def singleLearn(args, init_theta="random", out_theta=None):

    # path to store experimental data
    output_dir = os.path.join(project_dir, "experiments", args.experiment, "data_files")

    mdprank = MDPrankMain(args, init=init_theta)

    theta_new = mdprank.learn()

    if out_theta is not None:
        output_path = os.path.join(output_dir, out_theta)
        write_vector(theta_new, output_path)

    time0 = datetime.datetime.now()
    NDCG_mean = mdprank.eval(dataSet="validation")
    logging.info("Evaluation: averaged NDCG of validation set = %0.3f" % NDCG_mean)
    print("Evaluation: averaged NDCG of validation set = %0.3f" % NDCG_mean)
    logging.info("%ds used" % ((datetime.datetime.now() - time0).total_seconds()))
    print("%ds used" % ((datetime.datetime.now() - time0).total_seconds()))

    time0 = datetime.datetime.now()
    NDCG_mean = mdprank.eval(dataSet="test")
    logging.info("Evaluation: averaged NDCG of test set = %0.3f" % NDCG_mean)
    print("Evaluation: averaged NDCG of test set = %0.3f" % NDCG_mean)
    logging.info("%ds used" % ((datetime.datetime.now() - time0).total_seconds()))
    print("%ds used" % ((datetime.datetime.now() - time0).total_seconds()))

    return

def multiLearn(args, nLearner=1, init_theta="random", out_theta=None):
    # path to store experimental data
    output_dir = os.path.join(project_dir, "experiments", args.experiment, "data_files")

    learners = list()
    metric_valid = np.zeros(nLearner)
    metric_test = np.zeros(nLearner)

    theta_new_list = list()
    for i in range(nLearner):
        learners.append(MDPrankMain(args, init=init_theta, sample=i))
        theta_new = learners[i].learn()
        #output_path = output_dir + "theta1_" + str(i) + ".txt"
        #write_vector(theta_new, output_path)
        theta_new_list.append(theta_new)

        metric_valid[i] = learners[i].eval(dataSet="validation")
        metric_test[i] = learners[i].eval(dataSet="test")

    if out_theta is not None:
        output_path = os.path.join(output_dir, out_theta)
        write_vector(np.mean(np.array(theta_new_list), axis=0), output_path)

    logging.info("Evaluation: averaged metric of validation set = %0.3f" % np.mean(metric_valid))
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
