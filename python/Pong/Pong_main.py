#!/usr/bin/python
# -*- coding: <encoding name> -*-

import os, sys
import argparse
import time, datetime
import importlib
import numpy as np
import cv2
import torch
import collections
import gym
from tensorboardX import SummaryWriter

# path of the whole project
MDPrank_main_path = os.path.abspath(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(MDPrank_main_path)))
sys.path.append(project_dir)

from python.RLutils.algorithm.experience_buffer import Experience
from python.RLutils.algorithm.experience_buffer import ExperienceBuffer
#from python.Pong.environment.Pong_environment import PongEnvironment
from python.Pong.agent.PongAgent import PongAgent
#from python.Pong.agent.PongAgent1 import PongAgent1
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

    device = torch.device("cuda" if args.cuda else "cpu")

    env_name = hyperparams.ENVconfig['env_name']
    env = make_env(env_name)
    #env = PongEnvironment(hyperparams.ENVconfig)
    #env = PongEnvironment1(env_name)

    MEAN_REWARD_BOUND = 19.5

    GAMMA = 0.99
    BATCH_SIZE = 32
    REPLAY_SIZE = 10000
    LEARNING_RATE = 1e-4
    SYNC_TARGET_FRAMES = 1000
    REPLAY_START_SIZE = REPLAY_SIZE

    EPSILON_DECAY_LAST_FRAME = 10 ** 5
    EPSILON_START = 1.0
    EPSILON_FINAL = 0.02

    #net = DQN_nn(env.get_observation_space().shape, env.get_action_space().n).to(device)
    #tgt_net = DQN_nn(env.get_observation_space().shape, env.get_action_space().n).to(device)
    net = DQN_nn(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = DQN_nn(env.observation_space.shape, env.action_space.n).to(device)

    #net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    #tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + env_name)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)  #
    #agent = PongAgentOld(hyperparams.AGEconfig, env, buffer)
    #epsilon = EPSILON_START

    train = True
    if train:

        optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        total_rewards = []
        frame_idx = 0
        ts_frame = 0
        t0 = time.time()
        ts = t0
        best_mean_reward = None

        episode_reward = 0.0
        while True:
            frame_idx += 1
            epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

            #reward = agent.play_step(net, epsilon, device=device)

            action, new_state, reward, is_done = agent.play_step(net, epsilon, device=device)
            #exp = Experience(agent.state, action, reward, is_done, new_state)
            #buffer.append(exp)

            episode_reward += reward
            #if reward is not None:
            if is_done:
                total_rewards.append(episode_reward)
                episode_reward = 0.0
                speed = (frame_idx - ts_frame) / (time.time() -ts)
                ts = time.time()
                ts_frame = frame_idx
                mean_reward = np.mean(total_rewards[-100:])
                print("%d: done %d games, mean reward %.3f, nn param sum %.3f, eps %.2f, speed %.2f f/s, time passed %s" % (
                    frame_idx, len(total_rewards), mean_reward, net.get_param_sum(), epsilon,
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
                if mean_reward > hyperparams.ALGconfig['mean_reward_bound']:
                    print("Solved in %d frames!" % frame_idx)
                    break

            if len(buffer) < REPLAY_START_SIZE:
                continue

            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())

            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, net, tgt_net, device=device)
            loss_t.backward()
            optimizer.step()
        writer.close()

    else:
        visualize = True
        FPS = 25

        #print(project_dir + '/experiments/')
        #print(exp_dir)

        model_name = exp_dir + '/' + 'PongNoFrameskip-v4-best.dat'
        #print(model_name)
        #exit(3)

        net.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))

        state = env.reset()
        total_reward = 0.0
        c = collections.Counter()
        while True:
            start_ts = time.time()
            if visualize:
                env.render()

            action = agent.play(state, net, epsilon=0.0, device="cpu")

            c[action] += 1
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
            if visualize:
                delta = 1 / FPS - (time.time() - start_ts)
                if delta > 0:
                    time.sleep(delta)
        print("Total reward: %.2f" % total_reward)
        print("Action counts:", c)
        env.close()

    '''
    
    env_name = hyperparams.ENVconfig['env_name']
    env = PongEnvironment(hyperparams.ENVconfig)
    state = env.reset()

    #print(env.get_observation_space(), env.get_action_space())
    #print(env.get_observation_space().shape, env.get_action_space().n)

    
    net = DQN_nn(env.get_observation_space().shape, env.get_action_space().n).to(device)
    tgt_net = DQN_nn(env.get_observation_space().shape, env.get_action_space().n).to(device)
    
    print(net)

    exp_buffer = ExperienceBuffer(hyperparams.ALGconfig['replay_size'])
    #agent = PongAgent(hyperparams.AGEconfig, env)
    agent = PongAgent1(hyperparams.AGEconfig, env)
    #agent = PongAgentOld(hyperparams.AGEconfig, env, exp_buffer)

    #epsilon = hyperparams.AGEconfig['epsilon_start']

    optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams.AGEconfig['learning_rate'])
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts_start = time.time()
    ts = ts_start
    best_mean_reward = None


    max_episodes = 100
    episode_reward = 0
    while True:
        frame_idx += 1
        epsilon = max(hyperparams.AGEconfig['epsilon_final'], hyperparams.AGEconfig['epsilon_start'] - frame_idx / hyperparams.AGEconfig['epsilon_decay_last_frame'])

        exp, reward, is_done = agent.play(net, epsilon, device=device)
        #exp = Experience(self.state, action, reward, is_done, new_state)
        episode_reward += reward
        exp_buffer.append(exp)

        if is_done:

            total_rewards.append(episode_reward)

            #if len(total_rewards) >= 50:
            #    print('end, time: %f' % (time.time() - ts_start))
            #    exit(5)
            
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon, speed))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("total_reward", episode_reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), env_name + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > hyperparams.ALGconfig['mean_reward_bound']:
                print("Solved in %d frames!" % frame_idx)
                break
            agent.state = env.reset()
            episode_reward = 0.0

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
    
    '''

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

class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        #self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            #action = self.env.get_action_space().sample()
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        #self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            #done_reward = self.total_reward
            self._reset()
        return action, new_state, reward, is_done

    def play(self, state, net, epsilon=0.0, device="cpu"):
        if epsilon is not None and np.random.random() < epsilon:
            action = self.env.get_action_space().sample()
        else:
            state_a = np.array([state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

            '''
            state_v = torch.tensor(np.array([state], copy=False))
            q_vals = net(state_v).data.numpy()[0]
            action = np.argmax(q_vals)
            '''

        return action

'''
class DQN(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

    def get_param_sum(self):
        #for p in self.conv.parameters():
        #    print(p)
        #    break

        p_sum = 0.0
        for name, param in self.named_parameters():
            #print(name)
            #print(type(param))
            #print(param.size())
            #print(param.sum().item())
            p_sum += param.sum().item()
            #break

        #print(self.fc.weight)

        return p_sum
    
class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.get_action_space().sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

'''
class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class PongEnvironment1(object):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env = MaxAndSkipEnv(self.env)
        self.env = FireResetEnv(self.env)
        self.env = ProcessFrame84(self.env)
        self.env = ImageToPyTorch(self.env)
        self.env = BufferWrapper(self.env, 4)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        return


    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()
        return

    def get_observation_space(self):
        return self.env.observation_space

    def get_action_space(self):
        return self.env.action_space

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def close(self):
        self.env.close()
        return

def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)


if __name__ == "__main__":
    main()
