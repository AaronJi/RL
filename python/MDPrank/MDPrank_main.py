#!/usr/bin/python
# -*- coding: <encoding name> -*-

import imp
import os
import sys
import argparse
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
    def __init__(self, args, data=None):
        """
        Initialize MDPrank Main
        Args:
            args: arguments for experiment
            init: the init way of model parameter
            sample: the samling number for multi-learners
        """

        exp_name = args.experiment
        exp_dir = os.path.join(project_dir, 'experiments', exp_name)
        hyperparams_file = os.path.join(exp_dir, 'hyperparams.py')

        if args.silent:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s',
                                datefmt='%m-%d %H:%M',
                                filename=os.path.join(exp_dir, "exp.log"),
                                filemode='w')
        else:
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s',
                                datefmt='%m-%d %H:%M',
                                filename=os.path.join(exp_dir, "exp.log"),
                                filemode='w')

        output_dir = os.path.join(project_dir, "experiments", args.experiment, "data_files")
        if len(args.folder) > 0:
            output_dir = os.path.join(output_dir, args.folder)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        '''
        def append_sample_suffix(path, sample_suffix):
            if sample_suffix is None or len(sample_suffix) == 0:
                return path
            path_arg = path.split(',')
            if len(path_arg) <= 1:
                return path + sample_suffix
            else:
                return '.'.join(path_arg[:-1]) + sample_suffix + '.' + path_arg[-1]

        self.train_outputPath = os.path.join(output_dir, append_sample_suffix(args.train_output, self.sample))        
        '''
        self.train_outputPath = os.path.join(output_dir, args.train_output)
        self.valid_outputPath = os.path.join(output_dir, args.valid_output)
        self.test_outputPath = os.path.join(output_dir, args.test_output)

        self.hyperparams = imp.load_source('hyperparams', hyperparams_file)
        if args.silent:
            self.hyperparams.config['verbose'] = False
            self.hyperparams.ALGconfig['verbose'] = False

        ## init data
        self.datadealer = LectorDataDealer(self.hyperparams.DATAconfig)
        if data is None:
            nQuery_sample = None

            data_dir = os.path.join(project_dir, 'data')
            time0 = datetime.datetime.now()
            if len(args.training_set) > 0:
                trainingData = self.load_data(args.training_set, path_prefix=data_dir, nQuery_sample=nQuery_sample)
            else:
                trainingData = None
            if len(args.valid_set) > 0:
                validationData = self.load_data(args.valid_set, path_prefix=data_dir, nQuery_sample=nQuery_sample)
            else:
                validationData = None
            if len(args.test_set) > 0:
                testData = self.load_data(args.test_set, path_prefix=data_dir, nQuery_sample=nQuery_sample)
            else:
                testData = None
            print("data read, %0.2fs used" % ((datetime.datetime.now() - time0).total_seconds()))

            self.nTheta = self.datadealer.nFeature
        else:
            trainingData, validationData, testData = data
            self.nTheta = len(trainingData[trainingData.keys()[0]][trainingData[trainingData.keys()[0]].keys()[0]][0])

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

        logging.info("%d train data, %d validation data, %d test data" % (safe_len(trainingData), safe_len(validationData), safe_len(testData)))
        if self.hyperparams.config['verbose']:
            print("%d train data, %d validation data, %d test data" % (safe_len(trainingData), safe_len(validationData), safe_len(testData)))


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
        if args.param_init is None or args.param_init == "zero":
            pass
        elif args.param_init == "random":
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
                read_theta = os.path.join(output_dir, args.param_init)

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
        #if self.hyperparams.config['verbose']:
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

    def learn(self, batch_data=None, init_theta=None, nSamples=1):

        if batch_data is None:
            self.alg.learn(self.train_outputPath)
            out_theta = self.alg.agent.theta
        else:
            if init_theta is None:
                init_theta = self.alg.agent.theta
            out_theta = self.alg.batch_learn(init_theta, batch_data, self.train_outputPath, nSamples)

        logging.info("new param after learning: " + ','.join([str(th) for th in self.alg.agent.theta]))
        if self.hyperparams.config['verbose']:
            print("new param:")
            print(out_theta)

        return out_theta

    def eval(self, dataSet="test"):
        NDCG_mean, NDCG_queries = self.alg.eval(dataSet=dataSet)
        return NDCG_mean

    def predict_pointwise(self, dataSet="test"):
        predict_result = self.alg.predict_pointwise(dataSet=dataSet)
        return predict_result

def safe_len(data):
    if data is None:
        return 0
    else:
        return len(data)

def main():
    """ Main function to be run. """

    parser = argparse.ArgumentParser(description='Run the MDP rank algorithm.')

    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('--mode', type=str, default="train", help='study mode')

    parser.add_argument('--training_set', type=str, default="",
                        help='training set')
    parser.add_argument('--valid_set', type=str, default="",
                        help='validation set')
    parser.add_argument('--test_set', type=str, default="",
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
    parser.add_argument('--batch_size', type=str, default="-1", help="size of batch")
    parser.add_argument('--nLearner', type=str, default="1", help="number of learners")

    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')

    args = parser.parse_args()

    if args.mode == "train":
        try:
            batch_size = int(args.batch_size)
        except:
            batch_size = -1

        try:
            nLearner = int(args.nLearner)
        except:
            nLearner = 1

        if nLearner <= 0:
            nLearner = 1

        print("batch size = %d, number of learners = %d" % (batch_size, nLearner))
        learn(args, out_theta=args.param_out, batch_size=batch_size, nSamples=nLearner)

    if args.mode == "predict":
        pointwise_predict(args)

    return

def pointwise_predict(args):
    # path to store experimental data
    output_dir = os.path.join(project_dir, "experiments", args.experiment, "data_files")
    if len(args.folder) > 0:
        output_dir = os.path.join(output_dir, args.folder)

    # a dummy object in order to read data
    predictor = MDPrankMain(args)

    predict_result = predictor.predict_pointwise(dataSet="test")

    predict_result = ['\t'.join(result) for result in predict_result]
    output_path = os.path.join(output_dir, args.test_output)
    write_vector(predict_result, output_path)

    return

def learn(args, out_theta=None, batch_size=10, nSamples=1):
    # path to store experimental data
    output_dir = os.path.join(project_dir, "experiments", args.experiment, "data_files")
    if len(args.folder) > 0:
        output_dir = os.path.join(output_dir, args.folder)

    # a dummy object in order to read data
    dataReader = MDPrankMain(args)
    data = (dataReader.alg.env.data, dataReader.alg.env.validData, dataReader.alg.env.testData)
    dataReader.datadealer.set_batched_data(data[0], batch_size)

    mdprank = MDPrankMain(args, data=data)
    if batch_size is None or batch_size <= 0:
        theta_new = mdprank.learn()
    else:
        batch_theta = mdprank.alg.agent.theta
        for i in range(dataReader.datadealer.nBatch):
            batch_data = dataReader.datadealer.next_batch(batch_size)
            batch_theta = mdprank.learn(batch_data, batch_theta, nSamples=nSamples)
        theta_new = batch_theta

    if out_theta is not None:
        output_path = os.path.join(output_dir, out_theta)
        write_vector(theta_new, output_path)

    if len(args.valid_set) > 0:
        time0 = datetime.datetime.now()
        NDCG_mean = mdprank.eval(dataSet="validation")
        logging.info("Evaluation: averaged NDCG of validation set = %0.3f" % NDCG_mean)
        print("Evaluation: averaged NDCG of validation set = %0.3f" % NDCG_mean)
        logging.info("%ds used" % ((datetime.datetime.now() - time0).total_seconds()))
        print("%ds used" % ((datetime.datetime.now() - time0).total_seconds()))

    if len(args.test_set) > 0:
        time0 = datetime.datetime.now()
        NDCG_mean = mdprank.eval(dataSet="test")
        logging.info("Evaluation: averaged NDCG of test set = %0.3f" % NDCG_mean)
        print("Evaluation: averaged NDCG of test set = %0.3f" % NDCG_mean)
        logging.info("%ds used" % ((datetime.datetime.now() - time0).total_seconds()))
        print("%ds used" % ((datetime.datetime.now() - time0).total_seconds()))

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
