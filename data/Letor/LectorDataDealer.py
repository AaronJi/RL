#!/usr/bin/python
# -*- coding: <encoding name> -*-

from collections import defaultdict
import numpy as np
import re
import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

from DataDealer import DataDealer

class LectorDataDealer(DataDealer):

    def __init__(self, hyperparams):
        super(LectorDataDealer, self).__init__(hyperparams)
        self.nFeature = 0
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
                self.nFeature = K

            N = len(Mn)  # number of queries

            # load data
            data = dict()

            for line in lines:
                try:
                    label = int(line.split()[0])
                    qid = re.search(r"qid:([0-9]+).", line).group(1)
                    docid = line.strip().split("#docid = ")[1]

                    feature_vec = np.zeros(K)  # vector of features, for each query-document pair

                    feature_tup = [tuple.split(":") for tuple in line.strip().split("#")[0].strip().split()[2:]]
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
                    print("Unexpected error:", sys.exc_info()[0])

            # if required, normalize the feature vectors according to min and max of each query results
            if self._hyperparams['normalization']:
                for qid in data:
                    nDoc = len(data[qid])
                    tempFeature = np.zeros((nDoc, self.nFeature))
                    tempDoc = list()
                    tempLabel = list()
                    for i, docid in enumerate(data[qid]):
                        tempDoc.append(docid)
                        tempFeature[i] = data[qid][docid][0]
                        tempLabel.append(data[qid][docid][1])

                    newFeature = self.normalize_by_column(tempFeature)

                    # reconstruct the result
                    for i, docid in enumerate(tempDoc):
                        data[qid][docid] = (newFeature[i], tempLabel[i])
            return data

    def getPartData(self, data, nQuery, nNegDoc, nPosDoc):
        partial_data = {}

        queries = data.keys()

        for iq, query in enumerate(queries):
            if iq >= nQuery:
                break

            partial_data[query] = {}

            iNegDoc = 0
            iPosDoc = 0
            for doc in data[query].keys():
                if iPosDoc >= nPosDoc and iNegDoc >= nNegDoc:
                    break

                if iPosDoc < nPosDoc and data[query][doc][1] > 0:
                    iPosDoc += 1
                    partial_data[query][doc] = data[query][doc]

                if iNegDoc < nNegDoc and data[query][doc][1] <= 0:
                    iNegDoc += 1
                    partial_data[query][doc] = data[query][doc]

        return partial_data

def getQueries(data):
    return list(data.keys())

def getSearchDocs(data, qid):
    try:
        return list(data[qid].keys())
    except KeyError:
        return None


def getSearchData(data, qid):
    try:
        return [data[qid][docid] for docid in data[qid]]
    except KeyError:
        return None

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



