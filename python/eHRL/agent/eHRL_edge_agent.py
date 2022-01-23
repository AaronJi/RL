#!/usr/bin/python
# -*- coding: <encoding name> -*-

import os
import sys
import numpy as np
import tensorflow as tf
import logging

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)

from RLutils.agent.Agent import Agent

class eHRLEdgeAgent(Agent):

    def __init__(self, hyperparams):
        super(eHRLEdgeAgent, self).__init__(hyperparams)
        self.agent_name = 'edge'
        self.actor = None
        return

    def init(self, environment, batch_size):
        self._hyperparams['batch_size'] = batch_size
        self._hyperparams['user_space_size'] = environment.user_space_size
        self._hyperparams['item_space_size'] = environment.item_space_size
        self._hyperparams['state_space_size'] = environment.state_space_size
        self._hyperparams['action_space_size'] = environment.action_space_size
        self._hyperparams['item_sequence_len'] = environment._hyperparams['item_sequence_len']
        self._hyperparams['item_rec_len'] = environment._hyperparams['item_rec_len']
        self._hyperparams['emb_size'] = environment.emb_size
        self._hyperparams['use_user'] = environment._hyperparams['use_user']

        # '1: Initialize actor network f_θ^π and critic network Q(s, a|θ^µ) with random weights'
        self.actor = Actor(self._hyperparams, environment.item_embeddings, scope=self.agent_name + '_actor')
        return

    def get_recommendation_items(self, state, dict_embeddings, target=False, sess=None):
        '''
        Algorithm 2
        Args:
          item_rec_len: length of the recommendation list.
          state: current/remembered environment state (probably with noise).
          embeddings: Embeddings object. shape = (n_item, embedding_dim)
          target: boolean to use Actor's network or target network.
        Returns:
          Recommendation List: list of embedded items as future actions. shape = (batch_size, K, embedding_dim) => (batch_size, K*embedding_dim)
        '''
        actions = self.actor.act(state, target=target, sess=sess).reshape(self._hyperparams['item_rec_len'], 100)
        rec_items = []
        for action in actions:
            item = dict_embeddings[str(action)]
            rec_items.append(item)
        return rec_items

    def init_target_network(self, sess):
        sess.run(self.actor.init_target_network_params)
        return

    def update_target_network(self, sess):
        sess.run(self.actor.update_target_network_params)
        return


class Actor():
    ''' Policy function approximator. '''

    def __init__(self, hyperparams, item_embeddings, scope='actor'):
        self._hyperparams = hyperparams
        self.item_embeddings = item_embeddings
        self.scope = scope

        with tf.variable_scope(self.scope):
            # Build Actor network
            self.action_weights, self.state, self.sequence_length = self._build_net(self.scope + '_estimator')
            #self.network_params = tf.trainable_variables()
            self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + self.scope + '_estimator')

            # Build target Actor network
            self.target_action_weights, self.target_state, self.target_sequence_length = self._build_net(self.scope + '_target')
            #self.target_network_params = tf.trainable_variables()[len(self.network_params):]  # TODO: why sublist [len(x):]? Maybe because its equal to network_params + target_network_params
            self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + self.scope + '_target')

            # Initialize target network weights with network weights (θ^π′ ← θ^π)
            self.init_target_network_params = [self.target_network_params[i].assign(self.network_params[i]) for i in range(len(self.target_network_params))]

            # Update target network weights (θ^π′ ← τθ^π + (1 − τ)θ^π′)
            self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self._hyperparams['tau'], self.network_params[i]) + tf.multiply(1 - self._hyperparams['tau'], self.target_network_params[i])) for i in range(len(self.target_network_params))]

            # Gradient computation from Critic's action_gradients
            self.action_gradients = tf.placeholder(tf.float32, [None, self._hyperparams['action_space_size']])
            gradients = tf.gradients(tf.reshape(self.action_weights, [self._hyperparams['batch_size'], self._hyperparams['action_space_size']], name='42222222222'), self.network_params, self.action_gradients)
            params_gradients = list(map(lambda x: tf.div(x, self._hyperparams['batch_size'] * self._hyperparams['action_space_size']), gradients))

            # Compute ∇_a.Q(s, a|θ^µ).∇_θ^π.f_θ^π(s)
            self.optimizer = tf.train.AdamOptimizer(self._hyperparams['eta']).apply_gradients(zip(params_gradients, self.network_params))
        return

    def _build_net(self, scope):
        ''' Build the (target) Actor network. '''

        def gather_last_output(data, seq_lens):
            def cli_value(x, v):
                y = tf.constant(v, shape=x.get_shape(), dtype=tf.int64)
                x = tf.cast(x, tf.int64)
                return tf.where(tf.greater(x, y), x, y)

            batch_range = tf.range(tf.cast(tf.shape(data)[0], dtype=tf.int64), dtype=tf.int64)
            tmp_end = tf.map_fn(lambda x: cli_value(x, 0), seq_lens - 1, dtype=tf.int64)
            indices = tf.stack([batch_range, tmp_end], axis=1)
            return tf.gather_nd(data, indices)

        with tf.variable_scope(scope):
            # Inputs: current state, sequence_length
            # Outputs: action weights to compute the score Equation (6)
            state = tf.placeholder(tf.float32, [None, self._hyperparams['state_space_size']], 'state')
            user, item = tf.split(state, [self._hyperparams['user_space_size'], self._hyperparams['item_space_size']], axis=1)
            item_ = tf.reshape(item, [-1, self._hyperparams['item_sequence_len'], self._hyperparams['emb_size']])
            sequence_length = tf.placeholder(tf.int32, [None], 'sequence_length')
            cell = tf.nn.rnn_cell.GRUCell(self._hyperparams['emb_size'], activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal(), bias_initializer=tf.zeros_initializer())
            outputs, _ = tf.nn.dynamic_rnn(cell, item_, dtype=tf.float32, sequence_length=sequence_length)
            last_output = gather_last_output(outputs, sequence_length)  # TODO: replace by h
            if self._hyperparams['use_user']:
                input_tokens = [last_output, user]
                input_tensor = tf.concat(input_tokens, axis=-1)
            else:
                input_tensor = last_output
            x = tf.keras.layers.Dense(self._hyperparams['item_rec_len'] * self._hyperparams['emb_size'])(input_tensor)
            action_weights = tf.reshape(x, [-1, self._hyperparams['item_rec_len'], self._hyperparams['emb_size']])

        return action_weights, state, sequence_length

    def train(self, sess, state, sequence_length, action_gradients):
        '''  Compute ∇_a.Q(s, a|θ^µ).∇_θ^π.f_θ^π(s). '''
        sess.run(self.optimizer, feed_dict={self.state: state, self.sequence_length: sequence_length, self.action_gradients: action_gradients})
        return

    def predict(self, sess, state, sequence_length, target=False):
        if target:
            return sess.run(self.target_action_weights, feed_dict={self.target_state: state, self.target_sequence_length: sequence_length})
        else:
            return sess.run(self.action_weights, feed_dict={self.state: state, self.sequence_length: sequence_length})

    def act(self, state, target=False, sess=None):
        '''
        Algorithm 2
        Args:
          item_rec_len: length of the recommendation list.
          state: current/remembered environment state (probably with noise).
          embeddings: Embeddings object. shape = (n_item, embedding_dim)
          target: boolean to use Actor's network or target network.
        Returns:
          Recommendation List: list of embedded items as future actions. shape = (batch_size, K, embedding_dim) => (batch_size, K*embedding_dim)
        '''
        # '1: Generate w_t = {w_t^1, ..., w_t^K} according to Equation (5)'
        batch_size = state.shape[0]
        weights = self.predict(sess, state, [self._hyperparams['item_rec_len']] * batch_size, target)  # # array of (n_sample x n_feature_action x embeddding_dim)

        # '3: Score items in I according to Equation (6)'
        #cores = np.array([[[self.get_score(weights[i][k], embedding) for embedding in embeddings] for k in range(item_rec_len)] for i in range(batch_size)])
        s = []
        for i in range(batch_size):
            ss = []
            for k in range(self._hyperparams['item_rec_len']):
                sss = []
                for embedding in self.item_embeddings:
                    sss.append(self.get_score(weights[i][k], embedding))
                ss.append(sss)
            s.append(ss)
        scores = np.array(s)

        # '8: return a_t'
        a = []
        for i in range(batch_size):
            aa = []
            for k in range(self._hyperparams['item_rec_len']):
                aa.append(self.item_embeddings[np.argmax(scores[i][k])])
            a.append(aa)
        # TODO repeative issue?

        action = np.array(a)
        return action.reshape(batch_size, -1)

    def get_score(self, weights, embedding):
        '''
        Equation (6)
        Args:
          weights: w_t^k shape=(embedding_size,).
          embedding: e_i shape=(embedding_size,).
        Returns:
          score of the item i: score_i=w_t^k.e_i^T shape=(1,).
        '''
        ret = np.dot(weights, embedding.T)
        return ret
