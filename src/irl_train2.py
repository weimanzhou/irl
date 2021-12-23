'''
Author: your name
Date: 2021-12-12 11:45:17
LastEditTime: 2021-12-16 21:33:10
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \irl\src\irl_train.py
'''
import os
import gym
import tianshou as ts
import time
import argparse
from pettingzoo.mpe import simple_spread_v2

import torch, numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import BasePolicy, RandomPolicy, DQNPolicy, MultiAgentPolicyManager
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import A2CPolicy, ImitationPolicy
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic

from mlgridenv import MLGridEnv

env = MLGridEnv()


# print(dir(env))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9, help='a smaller gamma favors earlier win')
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--step-per-epoch', type=int, default=50000)
    parser.add_argument('--il-step-per-epoch', type=int, default=1000)
    parser.add_argument('--episode-per-collect', type=int, default=16)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument('--board-size', type=int, default=6)
    parser.add_argument('--win-size', type=int, default=4)
    parser.add_argument('--win-rate', type=float, default=0.9,
                        help='the expected winning rate')
    parser.add_argument('--watch', default=False, action='store_true',
                        help='no training, watch the play of pre-trained models')
    parser.add_argument('--agent-id', type=int, default=2,
                        help='the learned agent plays as the agent_id-th player. Choices are 1 and 2.')
    parser.add_argument('--resume-path', type=str, default='',
                        help='the path of agent pth file for resuming from a pre-trained agent')
    parser.add_argument('--opponent-path', type=str, default='',
                        help='the path of opponent agent pth file for resuming from a pre-trained agent')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    # a2c special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=None)
    parser.add_argument('--gae-lambda', type=float, default=1.)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    return parser.parse_args()


def get_agents(
        args=get_args(),
        agent_count=3,
        agent=None,  # BasePolicy
        optim=None,  # torch.optim.Optimizer
):  # return a tuple of (BasePolicy, torch.optim.Optimizer)

    # env = TicTacToeEnv(args.board_size, args.win_size)
    env = MLGridEnv()
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    net = Net(args.state_shape, args.action_shape,
              hidden_sizes=args.hidden_sizes, device=args.device).to(args.device)
    print(net)
    actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(net, device=args.device).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    dist = torch.distributions.Categorical
    # agent = A2CPolicy(
    #     actor,
    #     critic,
    #     optim,
    #     dist,
    #     discount_factor=args.gamma,
    #     gae_lambda=args.gae_lambda,
    #     vf_coef=args.vf_coef,
    #     ent_coef=args.ent_coef,
    #     max_grad_norm=args.max_grad_norm,
    #     reward_normalization=args.rew_norm,
    #     action_space=env.action_space,
    # )
    if args.resume_path:
        agent.load_state_dict(torch.load(args.resume_path))

    # if agent_opponent is None:
    #     if args.opponent_path:
    #         agent_opponent = deepcopy(agent_learn)
    #         agent_opponent.load_state_dict(torch.load(args.opponent_path))
    #     else:
    #         agent_opponent = RandomPolicy()

    agents = [A2CPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        reward_normalization=args.rew_norm,
        action_space=env.action_space,
    ) for i in range(agent_count)]
    agents = [DQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq
    ) for i in range(agent_count)]
    policy = MultiAgentPolicyManager(agents)
    return policy, optim


policy, optim = get_agents()
args = get_args()

train_env = MLGridEnv()
test_env = MLGridEnv()
buffer = VectorReplayBuffer(args.buffer_size, args.training_num)
train_collector = Collector(policy, train_env, buffer, exploration_noise=True)
test_collector = Collector(policy, test_env)

# ======== tensorboard logging setup =========
log_path = os.path.join(args.logdir, 'multi-agent', 'irl')
writer = SummaryWriter(log_path)
writer.add_text("args", str(args))
logger = TensorboardLogger(writer)


def save_fn(policy):
    if hasattr(args, 'model_save_path'):
        model_save_path = args.model_save_path
    else:
        model_save_path = os.path.join(
            args.logdir, 'multi-agent', 'irl', 'policy.pth')
    torch.save(
        policy.policies[args.agent_id - 1].state_dict(),
        model_save_path)


def stop_fn(mean_rewards):
    return mean_rewards >= env.spec.reward_threshold


def train_fn(epoch, env_step):
    policy.policies[args.agent_id - 1].set_eps(args.eps_train)


def test_fn(epoch, env_step):
    policy.policies[args.agent_id - 1].set_eps(args.eps_test)


# the reward is a vector, we need a scalar metric to monitor the training.
# we choose the reward of the learning agent
def reward_metric(rews):
    return rews[:, args.agent_id - 1]


train_data = train_collector.collect(10)

# start training, this may require about three minutes
# trainer
# result = onpolicy_trainer(
#     policy,
#     train_collector,
#     test_collector,
#     args.epoch,
#     args.step_per_epoch,
#     args.repeat_per_collect,
#     args.test_num,
#     args.batch_size,
#     episode_per_collect=args.episode_per_collect,
#     stop_fn=stop_fn,
#     save_fn=save_fn,
#     logger=logger
# )

# print(f'Finished training! Use {result["duration"]}')

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

torch.save(policy.state_dict(), 'irl.pth')
policy.load_state_dict(torch.load('irl.pth'))

policy.eval()
policy.set_eps(0.05)
collector = ts.data.Collector(policy, env, exploration_noise=True)
collector.collect(n_episode=1, render=1 / 35)
