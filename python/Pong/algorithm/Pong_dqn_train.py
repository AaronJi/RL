#!/usr/bin/python
# -*- coding: <encoding name> -*-


import time, datetime
import numpy as np
import torch


from python.RLutils.algorithm.experience_buffer import Experience
from python.RLutils.algorithm.experience_buffer import ExperienceBuffer

def dqn_train(hyperparams, env, agent, net, tgt_net, writer, exp_dir, device):
    buffer = ExperienceBuffer(hyperparams['replay_start_size'])

    env_name = env.env_name
    optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['learning_rate'])
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    t0 = time.time()
    ts = t0
    best_mean_reward = None

    state = env.reset()
    for i in range(hyperparams['max_episodes']):
        #state = env.reset()
        episode_reward = 0.0

        # episode starts
        while True:
            frame_idx += 1
            epsilon = agent.get_epsilon(frame_idx)

            action = agent.play(state, net, epsilon, device=device)
            new_state, reward, is_done, _ = env.step(action)
            exp = Experience(state, action, reward, is_done, new_state)
            buffer.append(exp)

            state = new_state

            episode_reward += reward
            if is_done:
                break

            if len(buffer) < hyperparams['replay_start_size']:
                continue

            if frame_idx % hyperparams['sync_target_frames'] == 0:
                tgt_net.load_state_dict(net.state_dict())

            optimizer.zero_grad()
            batch = buffer.sample(hyperparams['batch_size'])
            loss_t = calc_loss(batch, net, tgt_net, device=device)
            loss_t.backward()
            optimizer.step()

        # the episode ends
        total_rewards.append(episode_reward)

        speed = (frame_idx - ts_frame) / (time.time() - ts)
        ts = time.time()
        ts_frame = frame_idx
        mean_reward = np.mean(total_rewards[-100:])
        print(
            "%d: done %d games, last reward %.3f, mean reward %.3f, nn param sum %.3f, eps %.2f, speed %.2f f/s, time passed %s" % (
                frame_idx, len(total_rewards), episode_reward, mean_reward, net.get_param_sum(), epsilon,
                speed, datetime.timedelta(seconds=ts - t0)
            ))

        writer.add_scalar("epsilon", epsilon, frame_idx)
        writer.add_scalar("speed", speed, frame_idx)
        writer.add_scalar("reward_100", mean_reward, frame_idx)
        writer.add_scalar("reward", episode_reward, frame_idx)
        if best_mean_reward is None or best_mean_reward < mean_reward:
            torch.save(net.state_dict(), exp_dir + '/' + env_name + "-best.dat")
            if best_mean_reward is not None:
                print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
            best_mean_reward = mean_reward

        if mean_reward > hyperparams['mean_reward_bound']:
            print("Solved in %d frames!" % frame_idx)
            break

        state = env.reset()

    writer.close()

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    GAMMA = 0.99
    expected_state_action_values = rewards_v + GAMMA * next_state_values
    return torch.nn.MSELoss()(state_action_values, expected_state_action_values)