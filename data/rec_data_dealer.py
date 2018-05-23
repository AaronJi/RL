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
        if feature_schema is not None:
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
                sys.exit(1)
        return

    def load_data(self, data_path):
        # load data
        data = dict()
        try:
            with open(data_path, 'r') as file:
                lines = file.readlines()

                for line in lines:
                    uid, iid, label, feature_vec = self.parse_rec_line(line.rstrip('\n').rstrip('\r'))
                    if uid is None or iid is None or label is None or feature_vec is None:
                        continue
                    
                    # if there is an intercept in the linear model, add a dummy feature to match the intercept
                    if 'with_linear_intercept' in self._hyperparams and self._hyperparams['with_linear_intercept']:
                        feature_vec = np.hstack((feature_vec, np.array([1])))

                    if uid not in data:
                        udata = {iid: (feature_vec, label)}
                        data[uid] = udata
                    else:
                        data[uid][iid] = (feature_vec, label)

            file.close()
        except:
            print("error: data file can not read")
            sys.exit(1)

        # if there is an intercept in the linear model, add a dummy feature to match the intercept
        if 'with_linear_intercept' in self._hyperparams and self._hyperparams['with_linear_intercept']:
            self.nFeature += 1
            self.feature_names.append('dummy_intercept')

        # if required, normalize the feature vectors according to min and max of each query results
        if self._hyperparams['normalization']:
            data = self.data_normalize_by_column(data)

        return data

    def data_normalize_by_column(self, data):
        for uid in data:
            nItem = len(data[uid])
            tempFeature = np.zeros((nItem, self.nFeature))
            tempDoc = list()
            tempLabel = list()

            for i, iid in enumerate(data[uid]):
                tempDoc.append(iid)
                tempFeature[i] = data[uid][iid][0]
                tempLabel.append(data[uid][iid][1])

            newFeature = self.normalize_by_column(tempFeature)

            # reconstruct the result
            for i, iid in enumerate(tempDoc):
                data[uid][iid] = (newFeature[i], tempLabel[i])
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

                # -1 might be missing value; for linear model, just let then be zero?
                if vFeature < 0:
                    vFeature = 0

                feature_vec[iFeature] = vFeature
        except:
            print("warning: fail to parse the line: " + line)
            return None

        return uid, iid, label, feature_vec

    # generate a partial data for fast test purpose
    def getPartData(self, data, nQuery, nNegDoc=None, nPosDoc=None):
        partial_data = {}

        queries = data.keys()

        for iq, query in enumerate(queries):
            if iq >= nQuery:
                break

            partial_data[query] = {}

            if nNegDoc is None or nPosDoc is None:
                partial_data[query] = data[query]
            else:
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

    # if a query only contains neg labels, filter it
    def filter_all_neg(self, data_raw):
        data = dict()

        users = data_raw.keys()
        for uid in users:
            if self.query_has_pos(data_raw, uid):
                data[uid] = data_raw[uid]
        return data

    def set_batched_data(self, data, batch_size):
        self.data = data
        self.nData = len(data)
        self.query_list = list(data.keys())
        self.nBatch = int(np.floor(self.nData / batch_size))
        self.nBatch_sampled = 0
        self.current_batch_locator = 0

        return

    def next_batch(self, batch_size):
        if self.data is None:
            return None

        if self.current_batch_locator + batch_size >= self.nData:
            return None

        batch_query_list = self.query_list[self.current_batch_locator: self.current_batch_locator + batch_size]
        self.current_batch_locator += batch_size

        return {k: self.data[k] for k in batch_query_list}

    @staticmethod
    def getQueries(data):
        return list(data.keys())

    @staticmethod
    def getQeuryItems(data, uid):
        try:
            return list(data[uid].keys())
        except KeyError:
            return None

    @staticmethod
    def getQueryData(data, uid):
        try:
            return [data[uid][iid] for iid in data[uid]]
        except KeyError:
            return None

    @staticmethod
    def getQueryResult(data, uid):
        try:
            return data[uid]
        except KeyError:
            return None

    # for a given query, if there is at least one pos label
    @staticmethod
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
