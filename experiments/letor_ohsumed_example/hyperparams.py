""" Hyperparameters for LETOR ohsumed"""
from __future__ import division

DATAconfig = {
    'normalization': True
}

ALGconfig = {
    'method': "MDPrank",
    'eta': 0.01,  # learning rate
    'discount': 1.0,
    'iterations': 10, #100000
    'absErr': 1.0e-4,
    'nAbsErr': 3,
    'param_with_sigmoid': False,
    'verbose': True
}

AGEconfig = {
    "policyTYpe": "stochastic"
}

ENVconfig = {
    'reward_metric': "NDCG"
}

config = {
    'type': "rank",
    'verbose': True,
    'algorithm': ALGconfig,
    'agent': AGEconfig,
    'environment': ENVconfig
}

