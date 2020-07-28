import cv2
import gym
import gym.spaces
import numpy as np


#!/usr/bin/python
# -*- coding: <encoding name> -*-

from abc import ABCMeta, abstractmethod
import copy

from python.RLutils.environment.Environment import Environment

class GymEnvironment(Environment):
    """ Rank Environment superclass. """
    __metaclass__ = ABCMeta

    def __init__(self, hyperparams, env_name):
        super(GymEnvironment, self).__init__(hyperparams)
        self.env = gym.make(env_name)
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

    def test_train(self, n_episode, max_steps_episode):
        for i_episode in range(n_episode):
            observation = self.reset()
            for t in range(max_steps_episode):
                self.render()
                #print(observation)
                action = self.get_action_space().sample()
                observation, reward, done, info = self.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
        self.close()
        return



if __name__ == "__main__":

    for i, gym_env in enumerate(gym.envs.registry.all()):
        print(i, gym_env)

    #env_name = 'CartPole-v0'
    #env_name = 'MountainCar-v0'
    #env_name = 'MsPacman-v0'
    env_name=  'Hopper-v2'

    env = GymEnvironment({}, env_name)
    env.test_train(20, 1000)


    '''


    env = gym.make(env_name)
    env.reset()
    for _ in range(1000):
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())  # take a random action
    env.close()    

    for i_episode in range(1):
        observation = env.reset()
        for t in range(1000):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()    
    '''
