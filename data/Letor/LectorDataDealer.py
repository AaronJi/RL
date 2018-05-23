#!/usr/bin/python
# -*- coding: <encoding name> -*-

import numpy as np
import re
import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

from rec_data_dealer import  RecDataDealer
from DataDealer import DataDealer

class LectorDataDealer(RecDataDealer):

    def __init__(self, hyperparams):
        super(LectorDataDealer, self).__init__(hyperparams)
        self.data = None
        return

    # return a dict, whose key is query id, value is the querydata in a dict format
    # querydata's key is doc id; value is a tuple of feature vector and label
    def load_data(self, path):
        with open(path, "r") as f:
            lines = f.readlines()

            # determine dimensions
            K = self.nFeature  # number of features
            Mn = {}  # number of candicate documents of each query
            for line in lines:
                qid = re.search(r"qid:([0-9]+).", line).group(1)
                if qid not in Mn:
                    Mn[qid] = 1
                else:
                    Mn[qid] += 1
                nFeature = len(line.strip().split("#")[0].strip().split()[2:])
                K = np.max([K, nFeature])

            # update nFeature; if not the first time of loading (e.x. load testing data after loading training data),
            # check the consistency of features
            if self.nFeature > 0:
                assert K == self.nFeature
            else:
                # if there is an intercept in the linear model, add a dummy feature to match the intercept
                if 'with_linear_intercept' in self._hyperparams and self._hyperparams['with_linear_intercept']:
                    K += 1
                self.nFeature = K

            # load data
            data = dict()
            for line in lines:
                try:
                    label = int(line.split()[0])
                    qid = re.search(r"qid:([0-9]+).", line).group(1)
                    docid = line.strip().split("#docid = ")[1]

                    feature_vec = np.zeros(K)  # vector of features, for each query-document pair
                    if 'with_linear_intercept' in self._hyperparams and self._hyperparams['with_linear_intercept']:
                        #feature_vec = np.hstack((feature_vec, np.array([1])))
                        feature_vec[-1] = 1.0

                    feature_tup = [tup.split(":") for tup in line.strip().split("#")[0].strip().split()[2:]]
                    for tup in feature_tup:
                        index = int(tup[0]) - 1
                        x = float(tup[1])
                        feature_vec[index] = x

                    if qid not in data:
                        querydata = {docid: (feature_vec, label)}
                        data[qid] = querydata
                    else:
                        data[qid][docid] = (feature_vec, label)

                except:
                    print("Unexpected error: %s, line = %s" % (sys.exc_info()[0], line))

            # if required, normalize the feature vectors according to min and max of each query results
            if self._hyperparams['normalization']:
                data = self.data_normalize_by_column(data)

            return data

def getDocInfo(data, qid, docid):
    try:
        return data[qid][docid]
    except KeyError:
        return None

def getFeature(data, qid, docid):
    try:
        return data[qid][docid][0]
    except KeyError:
        return None

def getLabel(data, qid, docid):
    try:
        return data[qid][docid][1]
    except KeyError:
        return None


if __name__ == "__main__":
    lectordatadealer = LectorDataDealer({})
    path = "/Users/aaron/Projects/RL/data/Letor/TREC/TD2003/Data/All/TD2003.txt"

    data = lectordatadealer.load_data(path)

    print(data["1"])



