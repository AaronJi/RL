""" Hyperparameters for LETOR Trec example."""
from __future__ import division

DATAconfig = {
    'normalization': True
}

ALGconfig = {
    'method': "MDPrank",
    'eta': 0.005,  # learning rate
    'discount': 1.0,
    'iterations': 10, #100000
    'absErr': 1.0e-4,
    'nAbsErr': 3,
    'param_with_sigmoid': False,
    'verbose': True,
    'eval_valid_in_iters': True,
    'eval_test_in_iters': True,
    'fast_cal': True,
    "softmax_power": 1
}

AGEconfig = {
    "policyType": "stochastic",
    "softmax_power": 1
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

