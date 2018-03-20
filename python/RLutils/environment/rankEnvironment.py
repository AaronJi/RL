#!/usr/bin/python
# -*- coding: <encoding name> -*-

from abc import ABCMeta, abstractmethod
import copy

from .Environment import Environment

class RankEnvironment(Environment):
    """ Rank Environment superclass. """
    __metaclass__ = ABCMeta

    def __init__(self, hyperparams):
        super(RankEnvironment, self).__init__(hyperparams)
        return

    def transit(self, state, action):

        t = state[0]  # step t, i.e. the t-th position in the rank case
        Xt = state[1]  # the list of candidates which are not ranked yet

        assert 0 <= action < len(Xt)  # action is the index of candidate to rank next

        Xt_1 = copy.deepcopy(Xt)  # need to make copy or not?
        del Xt_1[action]
        state_new = (t+1, Xt_1)

        return state_new

    @abstractmethod
    def reward(self, state, action):
        raise NotImplementedError("Must be implemented in subclass.")


