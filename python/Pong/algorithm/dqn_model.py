import torch
#import torch.nn as nn

import numpy as np


class DQN_nn(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN_nn, self).__init__()

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

        return

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

    def calc_loss(self, batch, net, tgt_net, device="cpu"):
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
        expected_state_action_values = next_state_values * GAMMA + rewards_v
        return torch.nn.MSELoss()(state_action_values, expected_state_action_values)

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


if __name__ == "__main__":
    import gym
    import collections

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
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                    shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

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

    def make_env(env_name):
        env = gym.make(env_name)
        env = MaxAndSkipEnv(env)
        env = FireResetEnv(env)
        env = ProcessFrame84(env)
        env = ImageToPyTorch(env)
        env = BufferWrapper(env, 4)
        return ScaledFloatFrame(env)

    env = make_env('PongNoFrameskip-v4')
    device = torch.device("cpu")
    print(env.observation_space.shape, env.action_space.n)


    net = DQN_nn(env.observation_space.shape, env.action_space.n).to(device)
    p_sum = net.get_param_sum()
    print(p_sum)

