#!/usr/bin/env python
# coding: utf-8

import imp
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import argparse

# path of the whole project
main_path = os.path.abspath(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(main_path)))
sys.path.append(project_dir)


from python.LIRD.environment.MovieLensEnvironment import MovieLensEnvironment
from python.LIRD.algorithm.LIRD_algorithm import LIRDAlg
from python.LIRD.agent.LIRD_agent import LIRDAgent


class LIRDMain(object):

    """ Main class to run algorithms and experiments. """
    def __init__(self, args):
        """
        Initialize MDPrank Main
        Args:
            args: arguments for experiment
            init: the init way of model parameter
            sample: the samling number for multi-learners
        """
        exp_dir = os.path.join(project_dir, 'experiments', args.experiment)
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

        self.output_dir = os.path.join(project_dir, "experiments", args.experiment, "data_files")
        if len(args.folder) > 0:
            self.output_dir = os.path.join(self.output_dir, args.folder)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        def append_sample_suffix(path, sample_suffix):
            if sample_suffix is None or len(sample_suffix) == 0:
                return path
            path_arg = path.split(',')
            if len(path_arg) <= 1:
                return path + sample_suffix
            else:
                return '.'.join(path_arg[:-1]) + sample_suffix + '.' + path_arg[-1]

       # self.train_outputPath = os.path.join(self.output_dir, append_sample_suffix(args.train_output, self.sample))
        self.train_outputPath = os.path.join(self.output_dir, args.train_output)
        self.valid_outputPath = os.path.join(self.output_dir, args.valid_output)
        self.test_outputPath = os.path.join(self.output_dir, args.test_output)

        self.hyperparams = imp.load_source('hyperparams', hyperparams_file)
        if args.silent:
            self.hyperparams.config['verbose'] = False
            self.hyperparams.ALGconfig['verbose'] = False

        data_dir = os.path.join(project_dir, 'data')
        data_dir = os.path.join(data_dir, 'MovieLens')

        self.env = MovieLensEnvironment(self.hyperparams.ENVconfig)
        self.env.init(data_dir, self.hyperparams.DATAconfig)
        self.env.setTrainData()
        self.env.setTestData()

        # '1: Initialize actor network f_θ^π and critic network Q(s, a|θ^µ) with random weights'
        tf.reset_default_graph()  # For multiple consecutive execution
        self.sess = tf.Session()

        self.agent = LIRDAgent(self.hyperparams.AGEconfig)
        self.agent.init(self.env, self.hyperparams.ALGconfig['batch_size'])

        ## init algorithm module
        self.alg = LIRDAlg(self.hyperparams.ALGconfig)
        self.alg.initEnv(self.env)
        self.alg.initAgent(self.agent)

        logging.info("Init successfully")

        return

    def offline_learn(self):
        # Set up summary operators
        def build_summaries():
            episode_reward = tf.Variable(0.)
            tf.summary.scalar('reward', episode_reward)
            episode_max_Q = tf.Variable(0.)
            tf.summary.scalar('max_Q_value', episode_max_Q)
            critic_loss = tf.Variable(0.)
            tf.summary.scalar('critic_loss', critic_loss)

            summary_vars = [episode_reward, episode_max_Q, critic_loss]
            summary_ops = tf.summary.merge_all()
            return summary_ops, summary_vars

        summary_ops, summary_vars = build_summaries()
        self.sess.run(tf.global_variables_initializer())
        filename_summary = self.output_dir + '/summary.txt'
        writer = tf.summary.FileWriter(filename_summary, self.sess.graph)

        self.alg.train(self.sess, summary_ops, summary_vars, writer)

        writer.close()
        sess_path = self.output_dir + '/models.h5'
        tf.train.Saver().save(self.sess, sess_path, write_meta_graph=False)

        return


    def offline_test(self):
        # # Testing
        ratings, unknown, random_seen = self.alg.test_actor(self.env.train_users_data, target=False, n_session_len=10, sess=self.sess)
        print('%0.1f%% unknown' % (100 * unknown / (len(ratings) + unknown)))

        plt.figure(0)
        plt.subplot(1, 2, 1)
        plt.hist(ratings)
        plt.title('Predictions ; Mean = %.4f' % (np.mean(ratings)))
        plt.subplot(1, 2, 2)
        plt.hist(random_seen)
        plt.title('Random ; Mean = %.4f' % (np.mean(random_seen)))

        # self.env.test_users_data
        ratings, unknown, random_seen = self.alg.test_actor(self.env.train_users_data, target=True, n_session_len=10, sess=self.sess)
        print('%0.1f%% unknown' % (100 * unknown / (len(ratings) + unknown)))

        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.hist(ratings)
        plt.title('Predictions ; Mean = %.4f' % (np.mean(ratings)))
        plt.subplot(1, 2, 2)
        plt.hist(random_seen)
        plt.title('Random ; Mean = %.4f' % (np.mean(random_seen)))

        plt.show()

        return


def main():
    """ Main function to be run. """

    parser = argparse.ArgumentParser(description='Run the LIRD algorithm.')

    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('--mode', type=str, default="train", help='study mode')

    parser.add_argument('--training_set', type=str, default="", help='training set')
    parser.add_argument('--valid_set', type=str, default="", help='validation set')
    parser.add_argument('--test_set', type=str, default="", help='test set')

    parser.add_argument('--test_output', type=str, default="testoutput.txt", help='test output')
    parser.add_argument('--train_output', type=str, default="trainoutput.txt", help='train output')
    parser.add_argument('--valid_output', type=str, default="validoutput.txt", help='valid output')

    parser.add_argument('--param_init', type=str, default="random", help='param init')   # "zero", "random",  output_dir + "theta1.txt"
    parser.add_argument('--param_out', type=str, default="theta1.txt", help='param out')

    parser.add_argument('--folder', type=str, default="", help="folder of outputs")
    parser.add_argument('--batch_size', type=str, default="-1", help="size of batch")
    parser.add_argument('--nLearner', type=str, default="1", help="number of learners")

    parser.add_argument('-s', '--silent', action='store_true', help='silent debug print outs')

    args = parser.parse_args()


    # build data
    exp_name = args.experiment
    exp_dir = os.path.join(project_dir, 'experiments', exp_name)

    hyperparams_file = os.path.join(exp_dir, 'hyperparams.py')
    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    if args.silent:
        hyperparams.config['verbose'] = False

    hyperparams.ALGconfig['verbose'] = hyperparams.config['verbose']
    hyperparams.AGEconfig['verbose'] = hyperparams.config['verbose']
    hyperparams.ENVconfig['verbose'] = hyperparams.config['verbose']

    lird_rank = LIRDMain(args)

    #if args.mode == "train":
    lird_rank.offline_learn()
    #if args.mode == "predict":
    lird_rank.offline_test()

    return

if __name__ == "__main__":
    main()
