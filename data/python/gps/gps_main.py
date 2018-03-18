#!/usr/bin/python
# -*- coding: <encoding name> -*-

import logging
import imp
import os
import sys
import copy
import argparse
import threading
import time
import traceback


sys.path.append('/'.join(str.split(__file__, '/')[:-2]))

class GPSMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config, quit_on_end=False):
        """
        Initialize GPSMain
        Args:
            config: Hyperparameters for experiment
            quit_on_end: When true, quit automatically on completion
        """
        self._quit_on_end = quit_on_end

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str,
                        help='experiment name')
    parser.add_argument('-n', '--new', action='store_true',
                        help='create new experiment')
    parser.add_argument('-t', '--targetsetup', action='store_true',
                        help='run target setup')
    parser.add_argument('-r', '--resume', metavar='N', type=int,
                        help='resume training from iter N')
    parser.add_argument('-p', '--policy', metavar='N', type=int,
                        help='take N policy samples (for BADMM/MDGPS only)')
    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')
    parser.add_argument('-q', '--quit', action='store_true',
                        help='quit GUI automatically when finished')
    args = parser.parse_args()

    exp_name = args.experiment
    resume_training_itr = args.resume
    test_policy_N = args.policy

    print(args)

    gps_filepath = os.path.abspath(__file__)
    RL_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = RL_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'

    print(RL_dir)
    print(exp_dir)
    print(hyperparams_file)

    if args.silent:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    hyperparams = imp.load_source('hyperparams', hyperparams_file)

    print(hyperparams.algorithm['cost'])



if __name__ == "__main__":
    main()
