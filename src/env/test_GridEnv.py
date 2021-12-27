"""
Author: your name
Date: 2021-12-24 20:29:07
LastEditTime: 2021-12-24 22:22:33
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AEn
FilePath: \irl\src\env\test_GridEnv.py
"""
import sys

sys.path.extend(['E:\\GITHUB\\irl', 'E:/GITHUB/irl'])
from src.env.MAGridEnvWrapper import GridEnv

import tianshou as ts
from tianshou.policy import DQNPolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.utils.net.common import Net

import numpy as np
import torch

env = GridEnv()

observation_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n

net = Net(np.prod(observation_shape), np.prod(action_shape), hidden_sizes=[64, 64])
print(net)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)
policy = DQNPolicy(net, optim)

train_collector = Collector(policy, env, buffer=VectorReplayBuffer(10, 1), exploration_noise=True)
test_collector = Collector(policy, env)

onpolicy_trainer(
    policy,
    train_collector,
    test_collector=None,
    max_epoch=10,
    step_per_epoch=10,
    step_per_collect=4,
    episode_per_test=1,
    batch_size=6,
    repeat_per_collect=2
)

print('---------eval---------')

policy.eval()
policy.set_eps(0.05)
env = GridEnv()
collector = ts.data.Collector(policy, env, exploration_noise=True)
collector.collect(n_episode=10, render=1 / 35)

env.close()
