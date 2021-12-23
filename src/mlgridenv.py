'''
Author: your name
Date: 2021-12-12 11:46:36
LastEditTime: 2021-12-22 15:39:51
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \irl\src\mlgridenv.py
'''
from typing import Any, Dict, Tuple, Optional
import numpy as np

from tianshou.env import MultiAgentEnv
from pettingzoo.mpe import simple_spread_v2


class MLGridEnv(MultiAgentEnv):

    def __init__(self) -> None:
        super().__init__()

        self.env = simple_spread_v2.env()
        self.env.reset()
        self.action_space = self.env.env.action_space('agent_1')
        self.observation_space = self.env.state()
        # self.observation_space = self.env.env.observation_space('agent_1')

    def reset(self):
        self.env.reset()

        return {
            "agent_id": int(str(self.env.agent_selection).split('_')[1]) + 1,
            "obs": self.env.state(),
            "mask": True
        }

    def step(
            self, action: int
    ) -> Tuple[dict, np.ndarray, np.ndarray, dict]:

        _agent_selection = self.env.agent_selection
        agent_id = int(str(_agent_selection).split('_')[1]) + 1
        self.env.step(action)

        obs = {
            "agent_id": agent_id,
            # "obs": self.env.observation_space(_agent_selection),
            "obs": self.env.state(),
            "mask": True
        }
        # TODO 需要传入 agent, world, agents 参数
        # rewards = [self.env.reward(agent, world) for agent_id in agents]
        x_train1 = []
        for k in self.env.dones.items():
            x_train1.append(k[1])
        x_train2 = np.array(x_train1)
        rewards = []
        for k in self.env.rewards.items():
            rewards.append(k[1])
        rewards = np.array(rewards)
        # self.env.rewards[self.env.agent_selection]
        return obs, rewards, x_train2.all(), self.env.infos

    def render(self, mode='human'):
        self.env.render(mode='human')

if __name__ == '__main__':
    env = MLGridEnv()

    env.reset()
    print(env.observation_space.shape)
    
    obs, reward, done, info = env.step(1)

    print(obs['obs'].shape)
    

