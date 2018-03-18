#!/usr/bin/python
# -*- coding: <encoding name> -*-

from abc import ABCMeta, abstractmethod
import copy
import logging

from ENVconfig import ENVconfig

LOGGER = logging.getLogger(__name__)

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



