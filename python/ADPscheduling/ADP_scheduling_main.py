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

from data.Time_Space.time_space_data_dealer import TimeSpaceDataDealer
from environment.ADP_scheduling_environment import ADPschedulingEnvironment
from agent.ADP_scheduling_agent import ADP_scheduling_agent
from algorithm.ADP_scheduling_algorithm import ADP_scheduling_algorithm

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
    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    if args.silent:
        hyperparams.config['verbose'] = False

    hyperparams.ALGconfig['verbose'] = hyperparams.config['verbose']
    hyperparams.AGEconfig['verbose'] = hyperparams.config['verbose']
    hyperparams.ENVconfig['verbose'] = hyperparams.config['verbose']

    output_dir = os.path.join(project_dir, "experiments", args.experiment, "data_files")
    if len(args.folder) > 0:
        output_dir = os.path.join(output_dir, args.folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # set log
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

    # build data
    datadealer = TimeSpaceDataDealer(hyperparams.DATAconfig)
    random_seed = hyperparams.DATAconfig['random_seed']
    datadealer.generate_data(random_seed)
    ex_data_dir = project_dir + "/data/Time_Space/"
    datadealer.dump_data(ex_data_dir)
    time_space_info, init_resource, tasks, repositions = datadealer.load_data(ex_data_dir)

    print(time_space_info)

    T = len(time_space_info["time_detail"])  # number of time steps
    n = len(time_space_info["location_detail"])  # number of total resources
    nR = 0
    for location in init_resource:
        nR += init_resource[location]
    print("total %d time steps, %d locations, %d resources" % (T, n, nR))

    # build envrionment
    env = ADPschedulingEnvironment(hyperparams.ENVconfig, datadealer)
    env.set_environment_knowledge(time_space_info)

    # build agent
    agent = ADP_scheduling_agent(hyperparams.AGEconfig, T, n, nR, hyperparams.ALGconfig['max_period'])
    agent.set_environment_knowledge(time_space_info, repositions)

    # initialize
    alg = ADP_scheduling_algorithm(hyperparams.ALGconfig)
    alg.initEnv(env)
    alg.initAgent(agent)

    # offline train using data
    train_results = alg.offline_train((time_space_info, init_resource, tasks, repositions))

    # show results
    alg.show_results(train_results, time_space_info)

if __name__ == "__main__":
    main()



