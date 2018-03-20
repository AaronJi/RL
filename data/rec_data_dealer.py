#!/usr/bin/python
# -*- coding: <encoding name> -*-

import numpy as np
import os
import sys

parentdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parentdir)

from DataDealer import DataDealer

class RecDataDealer(DataDealer):

    def __init__(self, hyperparams, feature_schema=None, label_name="label"):
        super(RecDataDealer, self).__init__(hyperparams)
        self.nFeature = 0
        self.feature_names = list()
        self.init_schema(feature_schema)
        self.label_name = label_name
        return

    def init_schema(self, feature_schema):
        try:
            with open(feature_schema, 'r') as file:
                lines = file.readlines()

                for line in lines:
                    feature_name = line.rstrip('\n').rstrip('\r')
                    if len(feature_name) > 0 :
                        self.nFeature += 1
                        self.feature_names.append(feature_name)
            file.close()
        except:
            print("error: feature file can not read")
        return

    def load_data(self, data_path):
        # load data
        data = dict()

        try:
            with open(data_path, 'r') as file:
                lines = file.readlines()

                for line in lines:
                    uid, iid, label, feature_vec = self.parse_rec_line(line.rstrip('\n').rstrip('\r'))

                    if uid not in data:
                        udata = {iid: (feature_vec, label)}
                        data[uid] = udata
                    else:
                        data[uid][iid] = (feature_vec, label)
            file.close()
        except:
            print("error: data file can not read")

        return data

    # default line format: uid \t iid \t libsvm_data_line
    def parse_rec_line(self, line):
        arr = line.split('\t')

        if len(arr) < 3:
            return None

        uid = arr[0]
        iid = arr[1]

        tup_list = arr[2].split(' ')
        try:
            label = float(tup_list[0])

            # TODO: case without schema; probably with bug
            if self.nFeature <= 0:
                self.nFeature = len(tup_list) - 1

            feature_vec = np.zeros(self.nFeature)

            for tup in tup_list[1:]:
                iFeature = int(tup.split(':')[0])
                vFeature = float(tup.split(':')[1])

                feature_vec[iFeature] = vFeature
        except:
            return None

        return uid, iid, label, feature_vec

    # generate a partial data for fast test purpose
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

def getQueryData(data, uid):
    try:
        return [data[uid][iid] for iid in data[uid]]
    except KeyError:
        return None

# for a given query, if there is at least one pos label
def query_has_pos(data, uid):
    try:
        for iid in data[uid]:
            if data[uid][iid][1] > 0:
                return True
        return False

    except KeyError:
        return False

def main():
    pass

if __name__ == "__main__":
    main()
