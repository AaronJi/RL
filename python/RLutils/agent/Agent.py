#!/usr/bin/python
# -*- coding: <encoding name> -*-

from abc import ABCMeta, abstractmethod
import copy

from .AGEconfig import AGEconfig

class Agent(object):
    """ Environment superclass. """
    __metaclass__ = ABCMeta

    def __init__(self, hyperparams):
        config = copy.deepcopy(AGEconfig)
        config.update(hyperparams)
        self._hyperparams = config
        self.policy = None
        self.total_reward = None
        return

    @abstractmethod
    def act(self, state):
        raise NotImplementedError("Must be implemented in subclass.")
