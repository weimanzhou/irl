"""
Author: snowflake
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
from tianshou.policy import DQNPolicy, MultiAgentPolicyManager
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import Net

import numpy as np
import torch

gamma_1 = 0.9

env = GridEnv()


def get_agents():
    observation_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    env.reset()
    agent_count = env.agent_count
    agents = []
    for i in range(agent_count):
        net = Net(np.prod(observation_shape), np.prod(action_shape), hidden_sizes=[64, 64])
        optim = torch.optim.Adam(net.parametars(), lr=1e-3)
        policy = DQNPolicy(net, optim)
        agents.append(policy)
    return MultiAgentPolicyManager(agents)


train_envs = ts.env.DummyVectorEnv([lambda: GridEnv() for _ in range(10)])

policies = get_agents()

train_collector = Collector(policies, train_envs, buffer=VectorReplayBuffer(10000, 10), exploration_noise=True)
test_collector = Collector(policies, env)

np.random.seed(1)
torch.manual_seed(1)

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

writer = SummaryWriter('log/dqn')
logger = TensorboardLogger(writer)

onpolicy_trainer(
    policies,
    train_collector,
    test_collector=None,
    max_epoch=10,
    step_per_epoch=100,
    step_per_collect=10,
    episode_per_test=1,
    batch_size=64,
    repeat_per_collect=2,
    logger=logger
)

print('---------eval---------')

torch.save(policies.state_dict(), './model/sirl.pth')

policies.eval()
test_envs = ts.env.DummyVectorEnv([lambda: GridEnv() for _ in range(10)])
collector = ts.data.Collector(policies, test_envs, exploration_noise=True)
collector.collect(n_episode=2, render=1 / 35)

env.close()
