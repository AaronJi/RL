#!/usr/bin/python
# -*- coding: <encoding name> -*-

import os, sys
import argparse
import time
import importlib
import numpy as np
import torch
from tensorboardX import SummaryWriter

# path of the whole project
MDPrank_main_path = os.path.abspath(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(MDPrank_main_path)))
sys.path.append(project_dir)

from python.RLutils.algorithm.experience_buffer import Experience
from python.RLutils.algorithm.experience_buffer import ExperienceBuffer
from python.Pong.environment.Pong_environment import PongEnvironment
from python.Pong.agent.PongAgent import PongAgent
from python.Pong.agent.PongAgentOld import PongAgentOld
from python.Pong.algorithm.dqn_model import DQN_nn

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    """ Main function to be run. """
    # arguments
    parser = argparse.ArgumentParser(description='Run the Pong algorithm.')

    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('--folder', type=str, default="", help="folder of outputs")
    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")

    args = parser.parse_args()

    exp_name = args.experiment
    exp_dir = os.path.join(project_dir, 'experiments', exp_name)

    hyperparams_file = os.path.join(exp_dir, 'hyperparams.py')
    spec = importlib.util.spec_from_file_location('hyperparams', hyperparams_file)
    hyperparams = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hyperparams)

    if args.silent:
        hyperparams.config['verbose'] = False

    env_name = hyperparams.ENVconfig['env_name']
    env = PongEnvironment(hyperparams.ENVconfig)
    state = env.reset()

    #print(env.get_observation_space(), env.get_action_space())
    #print(env.get_observation_space().shape, env.get_action_space().n)

    device = torch.device("cuda" if args.cuda else "cpu")
    net = DQN_nn(env.get_observation_space().shape, env.get_action_space().n).to(device)
    tgt_net = DQN_nn(env.get_observation_space().shape, env.get_action_space().n).to(device)
    writer = SummaryWriter(comment="-" + env_name)
    print(net)

    exp_buffer = ExperienceBuffer(hyperparams.ALGconfig['replay_size'])
    #agent = PongAgent(hyperparams.AGEconfig, env)
    agent = PongAgentOld(hyperparams.AGEconfig, env, exp_buffer)

    #epsilon = hyperparams.AGEconfig['epsilon_start']

    optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams.AGEconfig['learning_rate'])
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts_start = time.time()
    ts = ts_start
    best_mean_reward = None


    '''
    max_episodes = 100
    for i in range(max_episodes):

        total_reward = 0.0
        while True:
            frame_idx += 1
            epsilon = max(hyperparams.AGEconfig['epsilon_final'], hyperparams.AGEconfig['epsilon_start'] - frame_idx / hyperparams.AGEconfig['epsilon_decay_last_frame'])

            #total_reward = agent.play_step(net, epsilon, device=device)
            action = agent.play(state, net, epsilon)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward

            exp = Experience(state, action, reward, is_done, new_state)
            exp_buffer.append(exp)
            state = new_state

            if is_done:
                # episode terminated
                state = env.reset()
                break

            if len(exp_buffer) < hyperparams.ALGconfig['replay_start_size']:
                continue

            if frame_idx % hyperparams.ALGconfig['sync_target_frames'] == 0:
                tgt_net.load_state_dict(net.state_dict())

            optimizer.zero_grad()
            batch = exp_buffer.sample(hyperparams.ALGconfig['batch_size'])
            loss_t = calc_loss(batch, net, tgt_net, device=device)
            loss_t.backward()
            optimizer.step()

        # when an episode ends
        total_rewards.append(total_reward)
        if len(total_rewards) >= 50:
            print('end, time: %f' % (time.time() - ts_start))
            exit(5)

        speed = (frame_idx - ts_frame) / (time.time() - ts)
        ts_frame = frame_idx
        ts = time.time()
        mean_reward = np.mean(total_rewards[-100:])
        print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (frame_idx, len(total_rewards), mean_reward, epsilon, speed))

        writer.add_scalar("epsilon", epsilon, frame_idx)
        writer.add_scalar("speed", speed, frame_idx)
        writer.add_scalar("reward_100", mean_reward, frame_idx)
        writer.add_scalar("total_reward", total_reward, frame_idx)

        if best_mean_reward is None or best_mean_reward < mean_reward:
            torch.save(net.state_dict(), env_name + "-best.dat")
            if best_mean_reward is not None:
                print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
            best_mean_reward = mean_reward

        if mean_reward > hyperparams.ALGconfig['mean_reward_bound']:
            print("Solved in %d frames!" % frame_idx)
            break
            
        #env.reset()
        #agent.total_reward = 0.0    
    
    '''


    while True:
        frame_idx += 1
        epsilon = max(hyperparams.AGEconfig['epsilon_final'], hyperparams.AGEconfig['epsilon_start'] - frame_idx / hyperparams.AGEconfig['epsilon_decay_last_frame'])

        total_reward = agent.play_step(net, epsilon, device=device)
        if total_reward is not None:
            total_rewards.append(total_reward)
            if len(total_rewards) >= 50:
                print('end, time: %f' % (time.time() - ts_start))
                exit(5)
            
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon, speed))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("total_reward", total_reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), env_name + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > hyperparams.ALGconfig['mean_reward_bound']:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(exp_buffer) < hyperparams.ALGconfig['replay_start_size']:
            continue

        if frame_idx % hyperparams.ALGconfig['sync_target_frames'] == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = exp_buffer.sample(hyperparams.ALGconfig['batch_size'])
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()       
    


    writer.close()

    #ts_end = time.time()
    #print(ts_end - ts_start)

    #env.test_train(1, 1000)
    return

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

if __name__ == "__main__":
    main()
