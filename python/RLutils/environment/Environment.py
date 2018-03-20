#!/usr/bin/python
# -*- coding: <encoding name> -*-

from abc import ABCMeta, abstractmethod
import copy

from .ENVconfig import ENVconfig

class Environment(object):
    """ Environment superclass. """
    __metaclass__ = ABCMeta

    def __init__(self, hyperparams):
        config = copy.deepcopy(ENVconfig)
        config.update(hyperparams)
        self._hyperparams = config
        return

    @abstractmethod
    def transit(self, state, action):
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def reward(self, state, action):
        raise NotImplementedError("Must be implemented in subclass.")



