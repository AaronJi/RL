import numpy as np
import time
import copy
import logging

from RLutils.algorithm.ALGconfig import ALGconfig
from python.RLutils.algorithm.experience_buffer import ReplayMemory
from python.RLutils.algorithm.actor_critic.DDPG import experience_replay

class LIRDAlg(object):

    def __init__(self, hyperparams):
        config = copy.deepcopy(ALGconfig)
        config.update(hyperparams)
        self._hyperparams = config

        #logging.info("learning rate = %f, discount rate = %f" % (self._hyperparams['eta'], self._hyperparams['discount']))
        #if self._hyperparams['verbose']:
        #    print("learning rate = %f, discount rate = %f" % (self._hyperparams['eta'], self._hyperparams['discount']))

        self.agent = None
        self.env = None
        return

    def initAgent(self, agent):
        self.agent = agent
        logging.info("init agent")
        return

    def initEnv(self, env):
        self.env = env
        logging.info("init environment")

        #self.item_rec_len = self.env._hyperparams['item_rec_len']
        return

    def train(self, sess, summary_ops, summary_vars, writer):

        # '2: Initialize target network f′ and Q′'
        self.agent.init_target_network(sess)

        # '3: Initialize the capacity of replay memory D'
        replay_memory = ReplayMemory(self._hyperparams['buffer_size'])  # Memory D in article
        replay = False

        start_time = time.time()
        for i_session in range(self._hyperparams['n_session']):  # '4: for session = 1, M do'
            session_reward = 0
            session_Q_value = 0
            session_critic_loss = 0

            # '5: Reset the item space I' is useless because unchanged.
            state = self.env.reset()  # '6: Initialize state s_0 from previous sessions'

            if (i_session + 1) % 10 == 0:  # Update average parameters every 10 episodes
                self.env.get_groups()

            for t in range(self._hyperparams['n_session_len']):  # '7: for t = 1, T do'
                # '8: Stage 1: Transition Generating Stage'

                # '9: Select an action a_t = {a_t^1, ..., a_t^K} according to Algorithm 2'
                action = self.agent.actor.act(state, sess=sess)

                # '10: Execute action a_t and observe the reward list {r_t^1, ..., r_t^K} for each item in a_t'
                reward, next_state = self.env.step(action)

                # '19: Store transition (s_t, a_t, r_t, s_t+1) in D'
                #experience = intract_experience(state.reshape(self.env.state_space_size), action.reshape(self.env.action_space_size), [reward], next_state.reshape(self.env.state_space_size))
                experience = intract_experience(state, action, [reward], next_state)
                replay_memory.add(experience)
                state = next_state  # '20: Set s_t = s_t+1'
                session_reward += reward
                #print(state.shape)
                # '21: Stage 2: Parameter Updating Stage'
                if replay_memory.size() >= self._hyperparams['batch_size']:  # Experience replay
                    replay = True
                    # '22: Sample minibatch of N transitions (s, a, r, s′) from D'
                    sampled_experiences = replay_memory.sample_batch(self._hyperparams['batch_size'])

                    states, actions, rewards, next_states = extract_experience(sampled_experiences)
                    replay_Q_value, loss_critic = experience_replay(states, actions, rewards, next_states, self.agent.actor, self.agent.critic, self._hyperparams['discount'], recurrent__length=self.env._hyperparams['item_rec_len'], sess=sess)
                    session_Q_value += replay_Q_value
                    session_critic_loss += loss_critic

                summary_str = sess.run(summary_ops, feed_dict={summary_vars[0]: session_reward, summary_vars[1]: session_Q_value, summary_vars[2]: session_critic_loss})

                writer.add_summary(summary_str, i_session)
                # print(state_to_items(self.sess, embeddings.embed(data['state'][0]), actor, item_rec_len, embeddings), state_to_items(self.sess, embeddings.embed(data['state'][0]), actor, item_rec_len, embeddings, True))

            str_loss = str('Loss=%0.4f' % session_critic_loss)
            print(('Episode %d/%d Reward=%d Time=%ds ' + (str_loss if replay else 'No replay')) % (
            i_session + 1, self._hyperparams['n_session'], session_reward, time.time() - start_time))
            start_time = time.time()
        return

    def test_actor(self, users_data, target=False, n_round=1, sess=None):
        ratings = []
        unknown = 0
        random_seen = []
        for i in range(n_round):
            print('*** test round %i, %i users***' %(i, len(users_data)))
            for user_data in users_data:
                user_id = user_data.iloc[0]['userId']
                user = self.env.datadealer.user_data[self.env.datadealer.user_data['userId'] == user_id].values

                historical_sampled_items = list(user_data.sample(self.env._hyperparams['item_sequence_len'])['itemId'])
                item = self.env.get_list_embedding(historical_sampled_items).reshape(1, -1)
                state = np.hstack((user, item))

                rec_items = self.agent.get_recommendation_items(state, self.env.dict_embeddings, target=target, sess=sess)
                for item in rec_items:
                    l = list(user_data.loc[user_data['itemId'] == item]['rating'])
                    assert (len(l) < 2)
                    if len(l) == 0:
                        unknown += 1
                    else:
                        ratings.append(l[0])
                for item in historical_sampled_items:
                    random_seen.append(list(user_data.loc[user_data['itemId'] == item]['rating'])[0])
        return ratings, unknown, random_seen

def intract_experience(state, action, reward, next_state):
    experience = [state, action, reward, next_state]
    return experience

def extract_experience(experiences):
    states = []
    actions = []
    rewards = []
    next_states = []
    for s in experiences:
        states.append(s[0].reshape(-1))
        actions.append(s[1].reshape(-1))
        rewards.append(s[2])
        next_states.append(s[3].reshape(-1))
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)

    return states, actions, rewards, next_states
