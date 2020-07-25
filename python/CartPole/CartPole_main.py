#!/usr/bin/python
# -*- coding: <encoding name> -*-

import os, sys
import argparse
#import imp
import importlib

# path of the whole project
MDPrank_main_path = os.path.abspath(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(MDPrank_main_path)))
sys.path.append(project_dir)

from python.CartPole.environment.CartPole_environment import CartPoleEnvironment

def main():
    """ Main function to be run. """

    # arguments
    parser = argparse.ArgumentParser(description='Run the MDP rank algorithm.')

    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('--folder', type=str, default="", help="folder of outputs")
    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')

    args = parser.parse_args()

    exp_name = args.experiment
    exp_dir = os.path.join(project_dir, 'experiments', exp_name)

    hyperparams_file = os.path.join(exp_dir, 'hyperparams.py')
    #hyperparams = imp.load_source('hyperparams', hyperparams_file)

    spec2 = importlib.util.spec_from_file_location('hyperparams', hyperparams_file)
    hyperparams = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(hyperparams)

    if args.silent:
        hyperparams.config['verbose'] = False

    print(hyperparams.ENVconfig)

    env = CartPoleEnvironment(hyperparams.ENVconfig)

    env.test_train(10, 1000)

    return

if __name__ == "__main__":
    main()
