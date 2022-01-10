import collections
import numpy as np
import random

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer(object):
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)


class ReplayMemory():
    ''' Replay memory D in article. '''

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        # self.buffer = [[row['state'], row['action'], row['reward'], row['n_state']] for _, row in data.iterrows()][-self.buffer_size:] TODO: empty or not?
        self.buffer = []

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        return

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return samples

