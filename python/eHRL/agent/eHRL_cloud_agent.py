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

class eHRLCloudAgent(Agent):

    def __init__(self, hyperparams):
        super(eHRLCloudAgent, self).__init__(hyperparams)
        self.agent_name = 'cloud'
        self.critic = None
        return

    def init(self, environment, batch_size):
        self._hyperparams['batch_size'] = batch_size
        self._hyperparams['user_space_size'] = environment.cloud_user_space_size
        self._hyperparams['item_space_size'] = environment.cloud_item_space_size
        self._hyperparams['state_space_size'] = environment.cloud_state_space_size
        self._hyperparams['action_space_size'] = environment.action_space_size
        self._hyperparams['item_sequence_len'] = environment._hyperparams['cloud_item_sequence_len']
        self._hyperparams['item_rec_len'] = environment._hyperparams['item_rec_len']
        self._hyperparams['emb_size'] = environment.emb_size
        self._hyperparams['use_user'] = environment._hyperparams['use_user']

        # '1: Initialize actor network f_θ^π and critic network Q(s, a|θ^µ) with random weights'
        self.critic = Critic(self._hyperparams, scope=self.agent_name + '_critic')
        return

    def init_target_network(self, sess):
        sess.run(self.critic.init_target_network_params)
        return

    def update_target_network(self, sess):
        sess.run(self.critic.update_target_network_params)
        return


class Critic():
    ''' Value function approximator. '''

    def __init__(self, hyperparams, scope='critic'):
        self._hyperparams = hyperparams
        self.scope = scope

        with tf.variable_scope(self.scope):
            # Build Critic network
            self.critic_Q_value, self.state, self.action, self.sequence_length = self._build_net(self.scope + '_estimator')
            self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + self.scope + '_estimator')

            # Build target Critic network
            self.target_Q_value, self.target_state, self.target_action, self.target_sequence_length = self._build_net(self.scope + '_target')
            self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + self.scope + '_target')

            # Initialize target network weights with network weights (θ^µ′ ← θ^µ)
            self.init_target_network_params = [self.target_network_params[i].assign(self.network_params[i]) for i in range(len(self.target_network_params))]

            # Update target network weights (θ^µ′ ← τθ^µ + (1 − τ)θ^µ′)
            self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self._hyperparams['tau'], self.network_params[i]) + tf.multiply(1 - self._hyperparams['tau'], self.target_network_params[i])) for i in range(len(self.target_network_params))]

            # Minimize MSE between Critic's and target Critic's outputed Q-values
            self.expected_reward = tf.placeholder(tf.float32, [None, 1])
            self.loss = tf.reduce_mean(tf.squared_difference(self.expected_reward, self.critic_Q_value))
            self.optimizer = tf.train.AdamOptimizer(10*self._hyperparams['eta']).minimize(self.loss)

            # Compute ∇_a.Q(s, a|θ^µ)
            self.action_gradients = tf.gradients(self.critic_Q_value, self.action)
            return

    def _build_net(self, scope):
        ''' Build the (target) Critic network. '''

        def gather_last_output(data, seq_lens):
            def cli_value(x, v):
                y = tf.constant(v, shape=x.get_shape(), dtype=tf.int64)
                return tf.where(tf.greater(x, y), x, y)

            this_range = tf.range(tf.cast(tf.shape(seq_lens)[0], dtype=tf.int64), dtype=tf.int64)
            tmp_end = tf.map_fn(lambda x: cli_value(x, 0), seq_lens - 1, dtype=tf.int64)
            indices = tf.stack([this_range, tmp_end], axis=1)
            return tf.gather_nd(data, indices)

        with tf.variable_scope(scope):
            # Inputs: current state, current action
            # Outputs: predicted Q-value
            state = tf.placeholder(tf.float32, [None, self._hyperparams['state_space_size']], 'state')
            user, item = tf.split(state, [self._hyperparams['user_space_size'], self._hyperparams['item_space_size']], axis=1)
            item_ = tf.reshape(item, [-1, self._hyperparams['item_sequence_len'], self._hyperparams['emb_size']])
            action = tf.placeholder(tf.float32, [None, self._hyperparams['action_space_size']], 'action')
            sequence_length = tf.placeholder(tf.int64, [None], name='critic_sequence_length')
            cell = tf.nn.rnn_cell.GRUCell(self._hyperparams['item_sequence_len'], activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal(), bias_initializer=tf.zeros_initializer())
            predicted_state, _ = tf.nn.dynamic_rnn(cell, item_, dtype=tf.float32, sequence_length=sequence_length)
            predicted_state = gather_last_output(predicted_state, sequence_length)

            input_tokens = [predicted_state, action]
            if self._hyperparams['use_user']:
                input_tokens.append(user)
            input_tensor = tf.concat(input_tokens, axis=-1)
            layer1 = tf.layers.Dense(32, activation=tf.nn.relu)(input_tensor)
            layer2 = tf.layers.Dense(16, activation=tf.nn.relu)(layer1)
            critic_Q_value = tf.layers.Dense(1)(layer2)
            return critic_Q_value, state, action, sequence_length

    def train(self, sess, state, action, sequence_length, expected_reward):
        ''' Minimize MSE between expected reward and target Critic's Q-value. '''
        return sess.run([self.critic_Q_value, self.loss, self.optimizer], feed_dict={self.state: state, self.action: action, self.sequence_length: sequence_length, self.expected_reward: expected_reward})

    def predict(self, sess, state, action, sequence_length, target=False):
        ''' Returns Critic's predicted Q-value. '''
        if target:
            return sess.run(self.target_Q_value, feed_dict={self.target_state: state, self.target_action: action, self.target_sequence_length: sequence_length})
        else:
            return sess.run(self.critic_Q_value, feed_dict={self.state: state, self.action: action, self.sequence_length: sequence_length})

    def get_action_gradients(self, sess, state, action, sequence_length):
        ''' Returns ∇_a.Q(s, a|θ^µ). '''
        return np.array(sess.run(self.action_gradients, feed_dict={self.state: state, self.action: action, self.sequence_length: sequence_length})[0])
