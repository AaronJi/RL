""" Hyperparameters for LETOR Trec example."""
from __future__ import division

DATAconfig = {
    'normalization': True,
    'with_linear_intercept': True
}

ALGconfig = {
    'method': "LIRD",
    'discount': 0.99,  # Gamma in Bellman equation
    'batch_size': 64,
    'n_session': 10,
    'n_session_len': 50,
    'buffer_size': 1000000,  # Size of replay memory D in article
}

AGEconfig = {
    "policyType": "stochastic",
    'tau': 0.001,  # τ in Algorithm 3
    'eta': 0.0001,  # learning rate
    "softmax_power": 1
}

ENVconfig = {
    'embedding_dim': 100,
    'reward_metric': "mean_ratings",
    'alpha': 0.5,  # α (alpha) in Equation (1),
    'gamma': 0.9,  # Γ (Gamma) in Equation (4)
    'fixed_length': True,  # Fixed memory length
    'item_sequence_len': 12,  # N in article
    'item_rec_len': 4,  # K in article
    'use_user': False,  # if use user features in the program
    'w_transit_noise': False  # if with noise in next state
}

config = {
    'type': "rank",
    'verbose': True,
    'algorithm': ALGconfig,
    'agent': AGEconfig,
    'environment': ENVconfig
}
