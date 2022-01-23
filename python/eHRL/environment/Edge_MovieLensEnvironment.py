#!/usr/bin/python
# -*- coding: <encoding name> -*-

import os
import sys
import copy
import logging
import numpy as np
import pandas as pd

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)

from data.MovieLens.MovieLensDataDealer import MovieLensDataDealer
from RLutils.environment.rankEnvironment import RankEnvironment
from RLutils.algorithm.simularity import cosine_simularity

class EHRLMovieLensEnvironment(RankEnvironment):

    def __init__(self, hyperparams):
        super(EHRLMovieLensEnvironment, self).__init__(hyperparams)
        self.alpha = self._hyperparams['alpha']  # α (alpha) in Equation (1)
        self.gamma = self._hyperparams['gamma']  # Γ (Gamma) in Equation (4)
        self.fixed_length = self._hyperparams['fixed_length']
        self.emb_size = self._hyperparams['embedding_dim']
        self.item_space_size = self.emb_size * self._hyperparams['item_sequence_len']
        self.action_space_size = self.emb_size * self._hyperparams['item_rec_len']
        return

    def init(self, data_dir, dataConfig):
        self.data_dir = data_dir

        ## init data
        self.datadealer = MovieLensDataDealer(dataConfig)
        self.datadealer.load_data(data_dir, self._hyperparams['item_sequence_len'], self._hyperparams['item_rec_len'])

        gen_embedding = False
        if gen_embedding:  # Generate embeddings?
            eg = EmbeddingsGenerator(self.datadealer.users_train, pd.read_csv(data_dir + '/ml-100k/u.data', sep='\t', names=['userId', 'itemId', 'rating', 'timestamp']), self._hyperparams['embedding_dim'])
            eg.train(nb_epochs=300)
            train_loss, train_accuracy = eg.test(self.datadealer.users_train)
            print('Train set: Loss=%.4f ; Accuracy=%.1f%%' % (train_loss, train_accuracy * 100))
            test_loss, test_accuracy = eg.test(self.datadealer.users_test)
            print('Test set: Loss=%.4f ; Accuracy=%.1f%%' % (test_loss, test_accuracy * 100))
            eg.save_embeddings(data_dir + '/ml-100k/embeddings.csv')

        self.load_embeddings(data_dir + '/embeddings.csv')
        assert self.get_embedding_size() == self.emb_size
        self.get_embedding_dict()

        self.user_space_size = len(self.datadealer.user_cols)
        self.state_space_size = self.item_space_size + self.user_space_size
        print('After init, embedding size %i, action space dim %i' % (self.emb_size, self.action_space_size))
        print('On the edge side, user_space_size %i, item_space_size dim %i, state_space_size dim %i' % (self.user_space_size, self.item_space_size, self.state_space_size))

        if self._hyperparams['with_cloud_latency']:
            self.cloud_user_space_size = self._hyperparams['cloud_user_space_size']
            self.cloud_item_space_size = self.emb_size * self._hyperparams['cloud_item_sequence_len']
            self.cloud_state_space_size = self.cloud_user_space_size + self.cloud_item_space_size
        else:
            self.cloud_user_space_size = self.user_space_size
            self.cloud_item_space_size = self.item_space_size
            self.cloud_state_space_size = self.state_space_size
        print('On the cloud  side, with edge-cloud transmission latency, user_space_size %i, item_space_size dim %i, state_space_size dim %i' % (self.cloud_user_space_size, self.cloud_item_space_size, self.cloud_state_space_size))
        return

    def setTrainData(self):
        self.train_users_data = self.datadealer.train_users_data
        train_data = self.datadealer.get_transition_from_file(self.data_dir + '/train.csv')
        self.embed_data(train_data)
        self.current_state = self.reset()
        self.groups = self.get_groups()
        return

    def setTestData(self):
        self.test_users_data = self.datadealer.test_users_data
        return

    def embed_data(self, data):
        merge_data = data.merge(self.datadealer.user_data, on='userId', how='left')
        states = []
        actions = []
        users = []
        for _, row in merge_data.iterrows():
            state = np.array([self.get_embedding(item_id) for item_id in row['state']])
            action = np.array([self.get_embedding(item_id) for item_id in row['action']])
            user = np.array([row[self.datadealer.user_cols].values])
            states.append(state)
            actions.append(action)
            users.append(user)

        self.embedded_data = pd.DataFrame()
        self.embedded_data['state'] = states
        self.embedded_data['action'] = actions
        self.embedded_data['user'] = users
        self.embedded_data['reward'] = merge_data['reward']  # n_sample * tuple of K rewards

        return

    def reset(self):
        n_data = len(self.embedded_data['state'])
        init_value = np.random.randint(0, n_data)
        user = self.embedded_data['user'][init_value]
        #self.init_state = self.embedded_data['state'].sample(1).values[0].reshape((1, -1))
        self.init_state = self.embedded_data['state'][init_value].reshape((1, -1))  # array of (n_feature_state x embeddding_dim) => (1 x n_feature_state * embeddding_dim)
        self.init_state = np.hstack((user, self.init_state))

        return self.init_state

    def step(self, action):
        '''
        Compute reward and update state.
        Args:
          actions: embedded chosen items.  # array of (n_feature_action x embeddding_dim)
        Returns:
          cumulated_reward: overall reward.
          current_state: updated state.  # array of (n_feature_state x embeddding_dim)
        '''

        # '18: Compute overall reward r_t according to Equation (4)'
        simulated_reward, cumulated_reward = self.simulate_rewards(self.current_state, action)

        current_items = self.current_state[:, self.user_space_size:]

        # '11: Set s_t+1 = s_t' <=> self.current_state = self.current_state
        action_multi_rows = action.reshape((self._hyperparams['item_rec_len'], -1))
        for k in range(len(simulated_reward)):  # '12: for k = 1, K do'
            if simulated_reward[k] > 0:  # '13: if r_t^k > 0 then'
                # '14: Add a_t^k to the end of s_t+1'
                current_state_multi_rows = current_items.reshape((self._hyperparams['item_sequence_len'], -1))
                current_state_multi_rows = np.append(current_state_multi_rows, [action_multi_rows[k]], axis=0)
                if self.fixed_length:  # '15: Remove the first item of s_t+1'
                    current_state_multi_rows = np.delete(current_state_multi_rows, 0, axis=0)
                current_items = current_state_multi_rows.reshape((1, -1))
        self.current_state[:, self.user_space_size:] = current_items

        if self._hyperparams['w_transit_noise']:
            exploration_noise = OrnsteinUhlenbeckNoise(self.state_space_size)
            self.current_state += exploration_noise.get().reshape(1, -1)

        return cumulated_reward, self.current_state

    def upload_cloud_state(self, edge_state):
        if self._hyperparams['with_cloud_latency']:
            cloud_state = np.hstack((edge_state[:, :self.cloud_user_space_size], edge_state[:, self.user_space_size:self.user_space_size+self.cloud_item_space_size]))
            #cloud_state = edge_state[:, :self.user_space_size + self.emb_size * self._hyperparams['cloud_item_sequence_len']]
        else:
            cloud_state = edge_state
        return cloud_state

    def get_groups(self):
        ''' Calculate average state/action value for each group. Equation (3). '''
        groups = []
        for rewards, group in self.embedded_data.groupby(['reward']):
            size = group.shape[0]
            states = np.array(list(group['state'].values))  # n_sample * array of (n_feature_state x embeddding_dim)
            actions = np.array(list(group['action'].values))  # n_sample * array of (n_feature_action x embeddding_dim)
            users = np.array(list(group['user'].values))

            group_stat = {
                'size': size,  # N_x in article
                'rewards': rewards,  # U_x in article (combination of rewards)
                'average state': (np.sum(states / np.linalg.norm(states, 2, axis=1)[:, np.newaxis], axis=0) / size).reshape((1, -1)),  # s_x^-
                'average action': (np.sum(actions / np.linalg.norm(actions, 2, axis=1)[:, np.newaxis], axis=0) / size).reshape((1, -1)),  # a_x^-
                'average user': np.mean(users, axis=0).reshape((1, -1))
            }
            groups.append(group_stat)

        return groups

    # Equation (1)
    def cosine_state_action(self, u_t, i_t, a_t, u_avg, i_avg, a_avg):
        cosine_simularity_item = cosine_simularity(i_t, i_avg)
        if self._hyperparams['use_user']:
            cosine_simularity_user = cosine_simularity(u_t, u_avg)
            #cosine_simularity_state = (cosine_simularity_item + cosine_simularity_user)/2.0
            cosine_simularity_state = (len(cosine_simularity_item)**2*cosine_simularity_item + len(cosine_simularity_user)**2*cosine_simularity_user) / (len(cosine_simularity_item)**2 + len(cosine_simularity_user)**2)
        else:
            cosine_simularity_state = cosine_simularity_item

        cosine_simularity_action = cosine_simularity(a_t, a_avg)
        cosine_simularity_state_action = (self.alpha * cosine_simularity_state + (1 - self.alpha) * cosine_simularity_action)

        return cosine_simularity_state_action.reshape((1,))

    def simulate_rewards(self, current_state, chosen_actions, reward_type='grouped cosine'):
        '''
        Calculate simulated rewards.
        Args:
          current_state: history, list of embedded items.  array of (1, n_feature_state * embeddding_dim)
          chosen_actions: embedded chosen items.  array of (1, n_feature_action * embeddding_dim)
          reward_type: from ['normal', 'grouped average', 'grouped cosine'].
        Returns:
          returned_rewards: most probable rewards.  tuple of (K, 1)
          cumulated_reward: probability weighted rewards.  scalar
        '''

        # TODO need a totally new array to have norm calculation?
        '''
        current_user = np.zeros((1, self.user_space_size), dtype=np.float)
        for i in range(1):
            for j in range(self.user_space_size):
                current_user[i][j] = current_state[i][j]
        current_items = np.zeros((1, self.item_space_size), dtype=np.float)
        for i in range(1):
            for j in range(self.item_space_size):
                current_items[i][j] = current_state[i][j+self.user_space_size]        
        '''
        current_user = current_state[:, :self.user_space_size]
        current_items = current_state[:, self.user_space_size:]

        if reward_type == 'normal':
            # Calculate simulated reward in normal way: Equation (2)
            probabilities = [self.cosine_state_action(current_user, current_items, chosen_actions, g['average user'], row['state'], row['action']) for _, row in self.embedded_data.iterrows()]
        elif reward_type == 'grouped average':
            # Calculate simulated reward by grouped average: Equation (3)
            probabilities = np.array([g['size'] for g in self.groups]) * [(self.alpha * (np.dot(current_items, g['average state'].T) / np.linalg.norm(current_state, 2)) + (1 - self.alpha) * (np.dot(chosen_actions, g['average action'].T) / np.linalg.norm(chosen_actions, 2))) for g in self.groups]
        elif reward_type == 'grouped cosine':
            # Calculate simulated reward by grouped cosine: Equations (1) and (3)
            probabilities = [self.cosine_state_action(current_user, current_items, chosen_actions, g['average user'], g['average state'], g['average action']) for g in self.groups]
        else:
            probabilities = np.array([[1]])  # dummy

        # Normalize (sum to 1)
        probabilities = np.array(probabilities) / sum(probabilities)

        # Get most probable rewards
        if reward_type == 'normal':
            returned_rewards = self.embedded_data.iloc[np.argmax(probabilities)]['reward']
        elif reward_type in ['grouped average', 'grouped cosine']:
            returned_rewards = self.groups[np.argmax(probabilities)]['rewards']

        # Equation (4)
        def overall_reward(rewards, gamma):
            return np.sum([(gamma ** k) * reward for k, reward in enumerate(rewards)])

        if reward_type in ['normal', 'grouped average']:
            # Get cumulated reward: Equation (4)
            cumulated_reward = overall_reward(returned_rewards, self.gamma)
        elif reward_type == 'grouped cosine':
            # Get probability weighted cumulated reward
            #cumulated_reward = np.sum([p * overall_reward(g['rewards'], self.gamma) for p, g in zip(probabilities, self.groups)])

            cumulated_reward_candidates = []
            for p, g in zip(probabilities, self.groups):
                cumulated_reward_candidates.append(p * overall_reward(g['rewards'], self.gamma))
            cumulated_reward = np.sum(cumulated_reward_candidates)

        return returned_rewards, cumulated_reward

    def load_embeddings(self, embeddings_path):
        ''' Load embeddings (a vector for each item). '''
        embeddings = pd.read_csv(embeddings_path, sep=';')
        item_embeddings = np.array([[np.float64(k) for k in e.split('|')] for e in embeddings['vectors']])
        self.item_embeddings = item_embeddings
        return

    def get_embedding_vector(self):
        return self.item_embeddings

    def get_embedding_size(self):
        return self.item_embeddings.shape[1]

    def get_embedding(self, item_index):
        return self.item_embeddings[item_index]

    def get_list_embedding(self, item_list):
        return np.array([self.get_embedding(item) for item in item_list])

    def get_embedding_dict(self):
        self.dict_embeddings = {}
        for i, item in enumerate(self.item_embeddings):
            str_item = str(item)
            assert (str_item not in self.dict_embeddings)
            self.dict_embeddings[str_item] = i
        return #dict_embeddings


class OrnsteinUhlenbeckNoise:
    ''' Noise for Actor predictions. '''

    def __init__(self, action_space_size, mu=0, theta=0.5, sigma=0.2):
        self.action_space_size = action_space_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_space_size) * self.mu

    def get(self):
        self.state += self.theta * (self.mu - self.state) + self.sigma * np.random.rand(self.action_space_size)
        return self.state


#import keras.backend as K
#from keras import Sequential
#from keras.layers import Dense, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

class EmbeddingsGenerator:
    def __init__(self, users, data, embedding_dim=100):
        self.users = users

        # preprocess
        self.data = data.sort_values(by=['timestamp'])
        # make them start at 0
        self.data['userId'] = self.data['userId'] - 1
        self.data['itemId'] = self.data['itemId'] - 1
        self.n_user = self.data['userId'].max() + 1
        self.n_item = self.data['itemId'].max() + 1
        self.user_item_pairs = {}  # list of rated movies by each user
        for userId in range(self.n_user):
            self.user_item_pairs[userId] = self.data[self.data.userId == userId]['itemId'].tolist()
        self.m = self.build_model(hidden_layer_size=embedding_dim)

    def build_model(self, hidden_layer_size=100):
        m = Sequential()
        m.add(Dense(hidden_layer_size, input_shape=(1, self.n_item)))
        m.add(Dropout(0.2))
        m.add(Dense(self.n_item, activation='softmax'))
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return m

    def train(self, nb_epochs=300, batch_size=10000):
        '''
        Trains the model from train_users's history
        '''
        for i in range(nb_epochs):
            print('%d/%d' % (i + 1, nb_epochs))
            batch = [self.generate_input(user_id=np.random.choice(self.users) - 1) for _ in range(batch_size)]
            X_train = np.array([b[0] for b in batch])
            y_train = np.array([b[1] for b in batch])
            self.m.fit(X_train, y_train, epochs=1, validation_split=0.5)

    def test(self, test_users, batch_size=100000):
        '''
        Returns [loss, accuracy] on the test set
        '''
        batch_test = [self.generate_input(user_id=np.random.choice(test_users) - 1) for _ in range(batch_size)]
        X_test = np.array([b[0] for b in batch_test])
        y_test = np.array([b[1] for b in batch_test])
        return self.m.evaluate(X_test, y_test)

    def generate_input(self, user_id):
        '''
        Returns a context and a target for the user_id
        context: user's history with one random movie removed
        target: id of random removed movie
        '''
        user_n_item = len(self.user_item_pairs[user_id])
        # picking random movie
        random_index = np.random.randint(0, user_n_item - 1)  # -1 avoids taking the last movie
        # setting target
        target = np.zeros((1, self.n_item))
        target[0][self.user_item_pairs[user_id][random_index]] = 1
        # setting context
        context = np.zeros((1, self.n_item))
        context[0][self.user_item_pairs[user_id][:random_index] + self.user_item_pairs[user_id][random_index + 1:]] = 1
        return context, target

    def save_embeddings(self, file_name):
        '''
        Generates a csv file containg the vector embedding for each movie.
        '''
        inp = self.m.input  # input placeholder
        outputs = [layer.output for layer in self.m.layers]  # all layer outputs
        functor = K.function([inp, K.learning_phase()], outputs)  # evaluation function

        # append embeddings to vectors
        vectors = []
        for movie_id in range(self.n_item):
            movie = np.zeros((1, 1, self.n_item))
            movie[0][0][movie_id] = 1
            layer_outs = functor([movie])
            vector = [str(v) for v in layer_outs[0][0][0]]
            vector = '|'.join(vector)
            vectors.append([movie_id, vector])

        # saves as a csv file
        embeddings = pd.DataFrame(vectors, columns=['item_id', 'vectors']).astype({'item_id': 'int32'})
        embeddings.to_csv(file_name, sep=';', index=False)
        return
