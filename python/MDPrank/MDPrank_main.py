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
    def __init__(self, args, init=None, data=None, sample=None):
        """
        Initialize MDPrank Main
        Args:
            args: arguments for experiment
            init: the init way of model parameter
            sample: the samling number for multi-learners
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

        output_dir = os.path.join(project_dir, "experiments", args.experiment, "data_files")
        if len(args.folder) > 0:
            output_dir = os.path.join(output_dir, args.folder)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        def append_sample_suffix(path, sample_suffix):
            if sample_suffix is None or len(sample_suffix) == 0:
                return path
            path_arg = path.split(',')
            if len(path_arg) <= 1:
                return path + sample_suffix
            else:
                return '.'.join(path_arg[:-1]) + sample_suffix + '.' + path_arg[-1]

        self.train_outputPath = os.path.join(output_dir, append_sample_suffix(args.train_output, self.sample))
        self.valid_outputPath = os.path.join(output_dir, append_sample_suffix(args.valid_output, self.sample))
        self.test_outputPath = os.path.join(output_dir, append_sample_suffix(args.test_output, self.sample))

        self.hyperparams = imp.load_source('hyperparams', hyperparams_file)
        if args.silent:
            self.hyperparams.config['verbose'] = False
            self.hyperparams.ALGconfig['verbose'] = False

        ## init data
        self.datadealer = LectorDataDealer(self.hyperparams.DATAconfig)
        if data is None:
            data_dir = os.path.join(project_dir, 'data')
            #data_dir = None
            time0 = datetime.datetime.now()
            trainingData = self.load_data(args.training_set, path_prefix=data_dir, nQuery_sample=None)
            validationData = self.load_data(args.training_set, path_prefix=data_dir, nQuery_sample=None)
            testData = self.load_data(args.training_set, path_prefix=data_dir, nQuery_sample=None)
            print("data read, %0.2fs used" % ((datetime.datetime.now() - time0).total_seconds()))

            self.nTheta = self.datadealer.nFeature
        else:
            trainingData, validationData, testData = data
            self.nTheta = len(trainingData[trainingData.keys()[0]][trainingData[trainingData.keys()[0]].keys()[0]][0])

        logging.info(
            "%d train data, %d validation data, %d test data" % (len(trainingData), len(validationData), len(testData)))
        if self.hyperparams.config['verbose']:
            print("%d train data, %d validation data, %d test data" % (len(trainingData), len(validationData), len(testData)))

        dump = False
        if dump:
            dump_dir = os.path.join(project_dir, 'data', args.experiment)
            if len(args.folder) > 0:
                dump_dir = os.path.join(dump_dir, args.folder)
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)

            self.datadealer.dump_pickle(data, os.path.join(trainingData, "trainingData.pkl"))
            self.datadealer.dump_pickle(data, os.path.join(validationData, "validationData.pkl"))
            self.datadealer.dump_pickle(data, os.path.join(testData, "testData.pkl"))

        ## init environment module

        env = MDPrankEnvironment(self.hyperparams.ENVconfig, self.datadealer)
        env.setTrainData(trainingData)
        env.setValidData(validationData)
        env.setTestData(testData)

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
                logging.warning("WARNING: fail to read initial parameter from file!")
                print("WARNING: fail to read initial parameter from file!")

        logging.info("init param: " + ','.join([str(th) for th in theta0]))
        if self.hyperparams.config['verbose']:
            print("init param:")
            print(theta0)

        agent = MDPrankAgent(self.hyperparams.AGEconfig, theta0)

        ## init algorithm module
        self.alg = MDPrankAlg(self.hyperparams.ALGconfig)
        self.alg.initEnv(env)
        self.alg.initAgent(agent)

        logging.info("Init successfully")

        return

    def load_data(self, data_path, path_prefix=None, nQuery_sample=None):
        if path_prefix is not None:
            data_path = os.path.join(path_prefix, data_path)


        if data_path.split('.')[-1] == 'pkl':
            data_raw = self.datadealer.load_pickle(data_path)
        else:
            data_raw = self.datadealer.load_data(data_path)

        if nQuery_sample is None:
            data = data_raw
        else:
            data = self.datadealer.getPartData(data_raw, nQuery_sample)

        return data

    def learn(self, batch_data=None, init_theta=None):
        train_outPath = self.train_outputPath + self.sample

        if batch_data is None:
            self.alg.learn(train_outPath)
        else:
            if init_theta is None:
                init_theta = self.alg.agent.theta
            self.alg.batch_learn(init_theta, batch_data, train_outPath)

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


    parser.add_argument('--folder', type=str, default="", help="folder of outputs")
    parser.add_argument('--nLearner', type=str, default="1", help="number of learners")

    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')

    args = parser.parse_args()

    try:
        nLearner = int(args.nLearner)
    except:
        nLearner = 1

    if nLearner > 1:
        multiLearn(args, nLearner=nLearner, init_theta=args.param_init, out_theta=args.param_out)
    else:
        #singleLearn(args, init_theta=args.param_init, out_theta=args.param_out)
        singleLearn(args, init_theta="random", out_theta=args.param_out, batch_size=10)
    return

def singleLearn1(args, init_theta="random", out_theta=None):

    # path to store experimental data
    output_dir = os.path.join(project_dir, "experiments", args.experiment, "data_files")
    if len(args.folder) > 0:
        output_dir = os.path.join(output_dir, args.folder)

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

def singleLearn(args, init_theta="random", out_theta=None, batch_size=10):
    # path to store experimental data
    output_dir = os.path.join(project_dir, "experiments", args.experiment, "data_files")
    if len(args.folder) > 0:
        output_dir = os.path.join(output_dir, args.folder)

    # a dummy object in order to read data
    dataReader = MDPrankMain(args, init=init_theta)
    data = (dataReader.alg.env.data, dataReader.alg.env.validData, dataReader.alg.env.testData)
    dataReader.datadealer.set_batched_data(data[0], batch_size)

    mdprank = MDPrankMain(args, init=init_theta, data=data)
    if batch_size is None:
        theta_new = mdprank.learn()
    else:
        batch_theta = mdprank.alg.agent.theta
        for i in range(dataReader.datadealer.nBatch):
            print("##### batch %d" % i)
            batch_data = dataReader.datadealer.next_batch(batch_size)
            batch_theta = mdprank.learn(batch_data, batch_theta)
        theta_new = batch_theta

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
    if len(args.folder) > 0:
        output_dir = os.path.join(output_dir, args.folder)

    # a dummy object in order to read data
    dataReader = MDPrankMain(args, init=init_theta)
    data = (dataReader.alg.env.data, dataReader.alg.env.validData, dataReader.alg.env.testData)

    learners = list()
    metric_valid = np.zeros(nLearner)
    metric_test = np.zeros(nLearner)

    theta_new_list = list()
    for i in range(nLearner):
        learners.append(MDPrankMain(args, init=init_theta, data=data, sample=i))
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
