'''
Author: your name
Date: 2021-12-24 15:05:21
LastEditTime: 2021-12-27 13:56:17
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \irl\src\env\MAGridEnvWrapper.py
'''
import sys, os

sys.path.extend(['E:\\GITHUB\\irl', 'E:/GITHUB/irl'])
import gym
import numpy as np
from src.env import EnvFindGoals


class GridEnv(gym.Env):

    def __init__(self):
        self.agent_count = None
        self.env = EnvFindGoals(agent_count=1, map_size=(10, 10), channel_range=2)

        self.observation_space = np.zeros(7)
        self.action_space = np.zeros(4)

    def step(self, action):
        state, rewards, done, info = self.env.step(action) \
            if isinstance(action, np.ndarray) else self.env.step(np.array([action]))

        obs = {
            "obs": state,
            "agent_id": self.env.agent_current + 1,
            "mask": np.full(4, True)
        }

        return obs, rewards, done, info

    def reset(self):
        states = self.env.reset()
        self.agent_count = self.env.agent_count
        return states
        # return {
        #     "obs": state,
        #     "agent_id": self.env.agent_current + 1,
        #     "mask": np.full(4, True)
        # }

    def render(self, mode="human"):
        self.env.render()

    def get_state(self, index):
        return self.env.get_state(index)

    def step_agent(self, idx, action):
        return self.env.step_agent(idx, action)

    def get_neighborhood(self, idx):
        return self.env.get_neighborhood(idx)


if __name__ == '__main__':
    env = GridEnv()
    env.reset()

    len_action = np.prod(env.action_space.shape)

    while True:
        state, reward, dones, info = env.step(np.random.randint(0, len_action, env.agent_count))
        print(state, reward)
        env.render()
