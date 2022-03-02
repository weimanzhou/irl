'''
Author: your name
Date: 2021-12-24 15:05:21
LastEditTime: 2021-12-27 13:56:17
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \irl\src\env\MAGridEnvWrapper.py
'''
import sys

sys.path.extend(['C:\\Users\\snowflake\\Documents\\GITHUB\\irl', 'C:/Users/snowflake/Documents/GITHUB/irl'])
import functools
import threading
import gym
import numpy as np
from src.env import EnvFindGoals


def synchronized(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)
    return wrapper


class GridEnv(gym.Env):

    def __init__(self):
        self.agent_count = 3
        self.env = EnvFindGoals(agent_count=self.agent_count, 
                                target_count=self.agent_count, 
                                map_size=(10, 10), 
                                channel_range=2)

        self.observation_space = np.zeros(7)
        self.action_space = np.zeros(5)

        self.lock = threading.Lock()

    # def step(self, action):
    #     state, rewards, done, info = self.env.step(action) \
    #         if isinstance(action, np.ndarray) else self.env.step(np.array([action]))

    #     obs = {
    #         "obs": state,
    #         "agent_id": self.env.agent_current + 1,
    #         "mask": np.full(4, True)
    #     }

    #     return obs, rewards, done, info

    def reset(self):
        states = self.env.reset()
        return states
        # return {
        #     "obs": state,
        #     "agent_id": self.env.agent_current + 1,
        #     "mask": np.full(4, True)
        # }

    def render(self, mode="human"):
        self.env.render()

    # def get_state(self, index):
    #     return self.env.get_state(index)

    @synchronized
    def step_agent(self, idx, action):
        return self.env.step_agent(idx, action)

    # def get_neighborhood(self, idx):
    #     return self.env.get_neighborhood(idx)




if __name__ == '__main__':
    env = GridEnv()
    env.reset()

    len_action = np.prod(env.action_space.shape)

    print('action count: {}'.format(len_action))

    def test1():
        for _ in range(4):
            state, reward, dones, info = env.step(np.random.randint(0, len_action, env.agent_count))
            print('test1')
    def test2():
        for _ in range(4):
            state, reward, dones, info = env.step(np.random.randint(0, len_action, env.agent_count))
            print('test2')
            
    threading.Thread(target=test1).start()
    threading.Thread(target=test2).start()

    # while True:
    #     state, reward, dones, info = env.step(np.random.randint(0, len_action, env.agent_count))
    #     print(state, reward)
    #     env.render()
