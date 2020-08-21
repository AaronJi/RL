#!/usr/bin/python
# -*- coding: <encoding name> -*-

import os, sys
import argparse
import time
import importlib
import torch
import collections
from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# path of the whole project
MDPrank_main_path = os.path.abspath(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(MDPrank_main_path)))
sys.path.append(project_dir)

from python.Pong.environment.Pong_environment import PongEnvironment
from python.Pong.agent.PongAgent import PongAgent
from python.Pong.algorithm.dqn_model import DQN_nn
from python.Pong.algorithm.Pong_dqn_train import dqn_train
from python.Pong.algorithm.Pong_dqn_play import dqn_play

def main():
    """ Main function to be run. """
    # arguments
    parser = argparse.ArgumentParser(description='Run the Pong algorithm.')

    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('--folder', type=str, default="", help="folder of outputs")
    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")

    args = parser.parse_args()

    exp_name = args.experiment
    exp_dir = os.path.join(project_dir, 'experiments', exp_name)

    hyperparams_file = os.path.join(exp_dir, 'hyperparams.py')
    spec = importlib.util.spec_from_file_location('hyperparams', hyperparams_file)
    hyperparams = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hyperparams)

    if args.silent:
        hyperparams.config['verbose'] = False

    device = torch.device("cuda" if args.cuda else "cpu")

    env_name = hyperparams.ENVconfig['env_name']
    env = PongEnvironment(hyperparams.ENVconfig)

    net = DQN_nn(env.get_observation_space().shape, env.get_action_space().n).to(device)
    tgt_net = DQN_nn(env.get_observation_space().shape, env.get_action_space().n).to(device)
    writer = SummaryWriter(comment="-" + env_name)
    print(net)

    agent = PongAgent(hyperparams.AGEconfig, env.get_action_space())

    train = True
    if train:
        dqn_train(hyperparams.ALGconfig, env, agent, net, tgt_net, writer, exp_dir, device)
    else:
        dqn_play(hyperparams.ALGconfig, env, agent, net, exp_dir)

    return


if __name__ == "__main__":
    main()
