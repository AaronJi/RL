#!/usr/bin/python
# -*- coding: <encoding name> -*-

'''
The dealer of data, for the case that all data coming from the offline data set
'''
import numpy as np
from abc import ABCMeta, abstractmethod
import copy
import pickle

from DATAconfig import DATAconfig

class DataDealer(object):
    __metaclass__ = ABCMeta

    def __init__(self, hyperparams):
        config = copy.deepcopy(DATAconfig)
        config.update(hyperparams)
        self._hyperparams = config
        return

    @abstractmethod
    def load_data(self, dataPath):
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def next_batch(self, batchsize):
        raise NotImplementedError("Must be implemented in subclass.")

    def dump_pickle(self, data, dumpFilePath):
        if data is None:
            return

        with open(dumpFilePath, "wb") as file:
            pickle.dump(data, file)
        file.close()

        return

    def load_pickle(self, picklePath):
        with open(picklePath, "rb") as file:
            try:
                data = pickle.load(file)
            except:
                data = None
        return data

    # normalization columnwisely; especially emphasized by the LETOR document
    def normalize_by_column(self, matrix):
        matrix = matrix.astype(float)
        max_minus_min = (matrix.max(axis=0) - matrix.min(axis=0))
        max_minus_min[max_minus_min == 0] = 1
        return (matrix - matrix.min(axis=0)) / max_minus_min

if __name__ == "__main__":
    datadealer = DataDealer(DATAconfig)

    A = np.array([[1, 4, 2], [3, 5, 2], [6, 2, 2]])
    print(A)

    print(datadealer.normalize_by_column(A))


    datadealer.dump_pickle(A, "d")

    B = datadealer.load_pickle("d")

    print(B)


