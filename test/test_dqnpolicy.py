import numpy as np
from tianshou.policy import DQNPolicy
from tianshou.data import Collector
from tianshou.data import Batch
from tianshou.utils.net.common import Net
from gym.envs.classic_control import CartPoleEnv

import torch

from src.mlgridenv import MLGridEnv

if __name__ == '__main__':
    env = CartPoleEnv()
    state_space = env.observation_space.shape or env.observation_space.n
    action_space = env.action_space.shape or env.action_space.n
    net = Net(state_shape=state_space, action_shape=action_space, hidden_sizes=[64, 64])
    print(net)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    target_update_freq = 100

    policy = DQNPolicy(
        net,
        optim,
        0.9,
        3,
        target_update_freq=100
    )

    collector1 = Collector(policy=policy, env=env)
    collector2 = Collector(policy=policy, env=CartPoleEnv())

    # 采样数据
    collector2.collect(n_step=100, random=True)

    policy(np.array([[0.11245313,  0.6052373 , -0.15940769, -1.1484663]]))

    # train_collector.collect(10)
    result = env.reset()
    result = Batch(obs=result['obs'], mask=result['mask'], info=None)
    action = policy(result)
    obs, rew, done, info = env.step(action)

    policy.update(10, collector2.buffer)

    d = Batch(
        obs=np.array([[0.11245313,  0.6052373, -0.15940769, -1.1484663]], dtype=np.float32),
        act=np.array([1]),
        rew=np.array([1]),
        done=np.array([False]),
        obs_next=np.array([[0.12455787,  0.412514, -0.18237701, -0.90971416]], dtype=np.float32),
        info={},
        policy=Batch(),
    )

