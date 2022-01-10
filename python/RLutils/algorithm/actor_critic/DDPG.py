import numpy as np

def experience_replay(states, actions, rewards, next_states, actor, critic, gamma, recurrent__length=None, sess=None):
    '''
    Experience replay.
    Args:
      experience: [states, actions, rewards, next_states].
      batch_size: sample size.
      actor: Actor network.
      critic: Critic network.
      embeddings: Embeddings object.
      state_space_size: dimension of states.
      action_space_size: dimensions of actions.
    Returns:
      Best Q-value, loss of Critic network for printing/recording purpose.
    '''
    batch_size = len(states)
    recurrent_tensor_size = [recurrent__length] * batch_size

    # '23: Generate a′ by target Actor network according to Algorithm 2'
    next_actions = actor.act(next_states, target=True, sess=sess)

    # Calculate predicted Q′(s′, a′|θ^µ′) value
    target_Q_value = critic.predict(sess, next_states, next_actions, recurrent_tensor_size, target=True)

    # '24: Set y = r + γQ′(s′, a′|θ^µ′)'
    expected_rewards = rewards + gamma * target_Q_value

    # '25: Update Critic by minimizing (y − Q(s, a|θ^µ))²'
    Q_value, loss_critic, _ = critic.train(sess, states, actions, recurrent_tensor_size, expected_rewards)

    # '26: Update the Actor using the sampled policy gradient'
    action_gradients = critic.get_action_gradients(sess, states, next_actions, recurrent_tensor_size)
    actor.train(sess, states, recurrent_tensor_size, action_gradients)

    # '27: Update the Critic target networks'
    sess.run(actor.update_target_network_params)

    # '28: Update the Actor target network'
    sess.run(critic.update_target_network_params)

    return np.amax(Q_value), loss_critic
