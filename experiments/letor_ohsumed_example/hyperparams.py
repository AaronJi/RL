""" Hyperparameters for LETOR ohsumed"""
from __future__ import division

DATAconfig = {
    'normalization': True,
    'with_linear_intercept': True
}

ALGconfig = {
    'method': "MDPrank",
    'eta': 0.0001,  # learning rate
    'discount': 1.0,
    'iterations': 10, #100000
    'nIter_batch': 10,
    'absErr': 1.0e-4,
    'nAbsErr': 3,
    'param_with_scale': 'minMax',  # minMax, sigmoid, None
    'update_by': 'batch',  # batch, episode, step
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

