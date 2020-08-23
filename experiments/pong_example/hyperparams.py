""" Hyperparameters for LETOR ohsumed"""
from __future__ import division

DATAconfig = {
}

ALGconfig = {
    'method': "Pong_play",
    'replay_size': 10000,
    'replay_start_size': 10000,
    'batch_size': 32,
    'sync_target_frames': 1000,
    'learning_rate': 1e-4,
    'max_episodes': 500,
    'mean_reward_bound': 19.0,
}

AGEconfig = {
    'policyType': "epsilon_greedy",
    'epsilon_start': 1.0,
    'epsilon_final': 0.02,
    'epsilon_decay_last_frame': 10**5,
}

ENVconfig = {
    'env_name': 'PongNoFrameskip-v4'
}

config = {
    'type': "rank",
    'verbose': True,
    'algorithm': ALGconfig,
    'agent': AGEconfig,
    'environment': ENVconfig
}
