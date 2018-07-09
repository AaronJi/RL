
""" Hyperparameters for LETOR ohsumed"""
from __future__ import division

DATAconfig = {
    'random_seed': 0,
    'distance_cal_type': 'euro',  # way of calculate two locations: european, ordinal

    # time
    'delta_ts': 1,  # time step interval
    'ts_start': 0,
    'ts_end': 23,
    'tau_max': 2,  # the upper bound of relocation time period

    # locations
    'location_shape': "rectangular",  # rectangular, hexagon
    'rectangular_size': (4, 4),

    # resources
    'resource_max': 10,  # max of initial number of resources

    # tasks
    'task_number_mean': [1, 0, 1, 1, 1, 2, 5, 10, 15, 9, 7, 6, 5, 5, 4, 5, 3, 3, 6, 8, 12, 15, 9, 2],  # mean of tasks as function of time
    'task_number_std': 1,  # std of tasks as function of time
    'task_income_mean': 5,  # mean of task incomes (per unit distance)
    'task_income_std': 1,  # std of task incomes (per unit distance)
    'task_speed': 1.0,  # speed of executing task

    # reposition
    'rep_cost': 0.5,  # reposition cost (per unit distance)
    'rep_speed': 1.0  # reposition speed

}

ALGconfig = {
    'nIter': 1000,  # max number of iterations
    'eta': 0.0001,  # learning rate
    'discount': 1.0,
    'max_period': 7,  # must be a positive int; if equaling to 1, the algorithm will decay to the single period mode
}

AGEconfig = {
    "policyType": "deterministic",
    'solver': 'ECOS',  # 'ECOS', 'ECOS_BB'
    'cave_step': 0.9,  # the step size of CAVE algorithm
    'cave_type': 'DUALMAX',  # DUALMAX or DUALNEXT; the slope update method of eq(15-16), Godfrey, Powell, 2002
}

ENVconfig = {
    'reward_metric': "NDCG"
}

config = {
    'type': "relocation",
    'verbose': True,
    'algorithm': ALGconfig,
    'agent': AGEconfig,
    'environment': ENVconfig,
}





