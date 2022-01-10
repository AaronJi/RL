#!/usr/bin/python
# -*- coding: <encoding name> -*-

import numpy as np
import re
import os
import sys
import pandas as pd
import random
import csv

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

from rec_data_dealer import RecDataDealer
from DataDealer import DataDealer


# 对age进行分段，映射成7组
def age_map(age):
    if age >= 1 and age <= 7: return 1
    if age >= 8 and age <=16: return 2
    if age >=17 and age <= 29: return 3
    if age >= 30 and age <= 39: return 4
    if age >= 40 and age <= 49: return 5
    if age >= 50 and age <= 59: return 6
    if age >= 60: return 7


#  occupation字段数值化
def occupations_map(occupation):
    occupations_dict = {'technician': 1,
     'other': 0,
     'writer': 2,
     'executive': 3,
     'administrator': 4,
     'student': 5,
     'lawyer': 6,
     'educator': 7,
     'scientist': 8,
     'entertainment': 9,
     'programmer': 10,
     'librarian': 11,
     'homemaker': 12,
     'artist': 13,
     'engineer': 14,
     'marketing': 15,
     'none': 16,
     'healthcare': 17,
     'retired': 18,
     'salesman': 19,
     'doctor': 20}
    return occupations_dict[occupation]

def zipcode_map(zipcode):
    try:
        return int(zipcode)/10000.0
    except:
        return 0


class MovieLensDataDealer(RecDataDealer):

    def __init__(self, hyperparams):
        super(MovieLensDataDealer, self).__init__(hyperparams)
        return

    def load_data(self, data_dir, item_sequence_len, item_rec_len):
        '''
        Load the data and merge the name of each movie.
        A row corresponds to a rate given by a user to a movie.

         Parameters
        ----------
        datapath :  string
                    path to the data 100k MovieLens
                    contains usersId;itemId;rating
        itempath :  string
                    path to the data 100k MovieLens
                    contains itemId;itemName
        Returns
        -------
        result :    DataFrame
                    Contains all the ratings
        '''
        data_path = data_dir + '/ml-100k/u.data'
        item_path = data_dir + '/ml-100k/u.item'
        user_path = data_dir + '/ml-100k/u.user'

        self.user_cols = ['userId', 'age', 'gender', 'occupation', 'zipcode']
        self.user_data = pd.read_csv(user_path, sep='|', names=self.user_cols)
        self.user_data['gender'] = self.user_data['gender'].map({'M': 1, 'F': 0})
        self.user_data['age'] = self.user_data['age'].apply(lambda age: age_map(age))
        self.user_data['occupation'] = self.user_data['occupation'].apply(lambda occupation: occupations_map(occupation))
        self.user_data['zipcode'] = self.user_data['zipcode'].apply(lambda zipcode: zipcode_map(zipcode))

        self.item_cols = ['itemId', 'itemName', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        self.item_data = pd.read_csv(item_path, sep='|', names=self.item_cols, encoding='latin-1') # ['itemId', 'itemName'] , usecols=range(2)

        interaction_data = pd.read_csv(data_path, sep='\t', names=['userId', 'itemId', 'rating', 'timestamp'])
        data = interaction_data

        self.users = data['userId'].unique()  # list of all users
        self.items = data['itemId'].unique()  # list of all items
        print('number of users %i, number of items %i' % (len(self.users), len(self.items)))

        self.historic_users_data = self.gen_histo(data)
        self.gen_train_test(0.8, seed=42)
        self.write_csv(data_dir + '/train.csv', self.train_users_data, n_states=[item_sequence_len], n_actions=[item_rec_len])
        self.write_csv(data_dir + '/test.csv', self.test_users_data, n_states=[item_sequence_len], n_actions=[item_rec_len])
        return

    def gen_histo(self, data):
        '''
        Group all rates given by users and store them from older to most recent.

        Returns
        -------
        result :    List(DataFrame)
                    List of the historic for each user
        '''
        historic_users_data = []
        for i, u in enumerate(self.users):
            temp = data[data['userId'] == u]
            temp = temp.sort_values('timestamp').reset_index()
            temp.drop('index', axis=1, inplace=True)
            historic_users_data.append(temp)
        return historic_users_data

    def gen_train_test(self, test_ratio, seed=None):
        '''
        Shuffle the historic of users and separate it in a train and a test set.
        Store the ids for each set.
        An user can't be in both set.

         Parameters
        ----------
        test_ratio :  float
                      Ratio to control the sizes of the sets
        seed       :  float
                      Seed on the shuffle
        '''
        n = len(self.historic_users_data)

        if seed is not None:
            random.Random(seed).shuffle(self.historic_users_data)
        else:
            random.shuffle(self.historic_users_data)

        self.train_users_data = self.historic_users_data[:int((test_ratio * n))]
        self.test_users_data = self.historic_users_data[int((test_ratio * n)):]
        self.users_train = [h.iloc[0, 0] for h in self.train_users_data]
        self.users_test = [h.iloc[0, 0] for h in self.test_users_data]
        return

    def get_transition_from_file(self, data_path):
        ''' Load data from train.csv or test.csv. '''

        data = pd.read_csv(data_path, sep=';')
        for col in ['state', 'n_state', 'action_reward']:
            data[col] = [np.array([[np.int(k) for k in ee.split('&')] for ee in e.split('|')]) for e in data[col]]

        for col in ['state', 'n_state']:
            data[col] = [np.array([e[0] for e in l]) for l in data[col]]

        data['action'] = [[e[0] for e in l] for l in data['action_reward']]
        data['reward'] = [tuple(e[1] for e in l) for l in data['action_reward']]

        data.drop(columns=['action_reward'], inplace=True)

        return data


    def sample_histo(self, user_histo, action_ratio=0.8, max_samp_by_user=5, max_state=100, max_action=50, n_states=[], n_actions=[]):
        '''
        For a given historic, make one or multiple sampling.
        If no optional argument given for nb_states and nb_actions, then the sampling
        is random and each sample can have differents size for action and state.
        To normalize sampling we need to give list of the numbers of states and actions
        to be sampled.

        Parameters
        ----------
        user_histo :  DataFrame
                          historic of user
        delimiter :       string, optional
                          delimiter for the csv
        action_ratio :    float, optional
                          ratio form which movies in history will be selected
        max_samp_by_user: int, optional
                          Nulber max of sample to make by user
        max_state :       int, optional
                          Number max of movies to take for the 'state' column
        max_action :      int, optional
                          Number max of movies to take for the 'action' action
        n_states :       array(int), optional
                          Numbers of movies to be taken for each sample made on user's historic
        n_actions :      array(int), optional
                          Numbers of rating to be taken for each sample made on user's historic

        Returns
        -------
        states :         List(String)
                         All the states sampled, format of a sample: itemId&rating
        actions :        List(String)
                         All the actions sampled, format of a sample: itemId&rating


        Notes
        -----
        States must be before(timestamp<) the actions.
        If given, size of nb_states is the numbller of sample by user
        sizes of nb_states and nb_actions must be equals
        '''
        user_id = user_histo.iloc[0, 0]

        n = len(user_histo)
        sep = int(action_ratio * n)
        nb_sample = random.randint(1, max_samp_by_user)
        if not n_states:
            n_states = [min(random.randint(1, sep), max_state) for i in range(nb_sample)]
        if not n_actions:
            n_actions = [min(random.randint(1, n - sep), max_action) for i in range(nb_sample)]
        assert len(n_states) == len(n_actions), 'Given array must have the same size'

        states = [user_id]
        actions = []
        # SELECT SAMPLES IN HISTO
        for i in range(len(n_states)):
            sample_states = user_histo.iloc[0:sep].sample(n_states[i])
            sample_actions = user_histo.iloc[-(n - sep):].sample(n_actions[i])

            sample_state = []
            sample_action = []
            for j in range(n_states[i]):
                row = sample_states.iloc[j]
                # FORMAT STATE
                state = str(row.loc['itemId']) + '&' + str(row.loc['rating'])
                sample_state.append(state)

            for j in range(n_actions[i]):
                row = sample_actions.iloc[j]
                # FORMAT ACTION
                action = str(row.loc['itemId']) + '&' + str(row.loc['rating'])
                sample_action.append(action)

            states.append(sample_state)
            actions.append(sample_action)

        return states, actions

    def write_csv(self, filename, histo_to_write, delimiter=';', action_ratio=0.8, max_samp_by_user=5, max_state=100, max_action=50, n_states=[], n_actions=[]):
        '''
        From  a given historic, create a csv file with the format:
        columns : state;action_reward;n_state
        rows    : itemid&rating1 | itemid&rating2 | ... ; itemid&rating3 | ... | itemid&rating4; itemid&rating1 | itemid&rating2 | itemid&rating3 | ... | item&rating4
        at filename location.

        Parameters
        ----------
        filename :        string
                          path to the file to be produced
        histo_to_write :  List(DataFrame)
                          List of the historic for each user
        delimiter :       string, optional
                          delimiter for the csv
        action_ratio :    float, optional
                          ratio form which movies in history will be selected
        max_samp_by_user: int, optional
                          Nulber max of sample to make by user
        max_state :       int, optional
                          Number max of movies to take for the 'state' column
        max_action :      int, optional
                          Number max of movies to take for the 'action' action
        n_states :       array(int), optional
                          Numbers of movies to be taken for each sample made on user's historic
        n_actions :      array(int), optional
                          Numbers of rating to be taken for each sample made on user's historic

        Notes
        -----
        if given, size of nb_states is the numbller of sample by user
        sizes of nb_states and nb_actions must be equals

        '''
        with open(filename, mode='w') as file:
            f_writer = csv.writer(file, delimiter=delimiter)
            f_writer.writerow(['userId', 'state', 'action_reward', 'n_state'])
            for user_histo in histo_to_write:
                states, actions = self.sample_histo(user_histo, action_ratio, max_samp_by_user, max_state, max_action, n_states, n_actions)

                user_id_str = states[0]
                # FORMAT STATE
                state_str = '|'.join(states[1])
                # FORMAT ACTION
                action_str = '|'.join(actions[0])
                # FORMAT N_STATE
                n_state_str = state_str + '|' + action_str

                #print(user_id_str, state_str, action_str, n_state_str)
                f_writer.writerow([user_id_str, state_str, action_str, n_state_str])

                '''
                for i in range(len(states)):
                    # FORMAT STATE
                    state_str = '|'.join(states[i])
                    # FORMAT ACTION
                    action_str = '|'.join(actions[i])
                    # FORMAT N_STATE
                    n_state_str = state_str + '|' + action_str

                    f_writer.writerow([state_str, action_str, n_state_str])                
                '''
        return
