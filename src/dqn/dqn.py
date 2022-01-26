import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import collections

import time

from gym_setup import *


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        # (1, 84, 84, 1)
        return int(np.prod(self.conv(torch.zeros(1, *shape)).size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


DEFAULT_ENV_NAME = 'PongNoFrameskip-v4'
MEAN_REWARD_ROUND = 19

GAMMA = 0.9
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

experience_template = collections.namedtuple(
    'experience',
    field_names=['state', 'action', 'reward', 'done', 'state_']
)


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        states, actions, rewards, dones, states_ = \
            zip(*[self.buffer[idx] for idx in indices])

        return np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(states_)


def calculate_loss(batch, net, target_net):
    states, actions, rewards, dones, states_ = batch

    states_v = torch.tensor(np.array(states, copy=False))
    states_v_ = torch.tensor(np.array(states_, copy=False))
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    done_mask = torch.BoolTensor(dones)

    state_action_v = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        state_values_ = target_net(states_v_).max(1)[0]
        state_values_[done_mask] = 0.0
        state_values_ = state_values_.detach()

    expected_state_action_values = rewards_v + GAMMA * state_values_

    return nn.MSELoss()(state_action_v, expected_state_action_values)


class Agent:
    def __init__(self, env, exp_buffer):
        super(Agent, self).__init__()

        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0):
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            # 这几行代码是为了找到产生最大值的 动作 a
            state_v = torch.tensor(np.array([self.state], copy=False))
            q_val_v = net(state_v)
            _, act_v = torch.max(q_val_v, dim=1)
            action = int(act_v.item())

        state_, reward, done, info = self.env.step(action)
        self.total_reward += reward

        # 存储 agent 的经验
        exp = experience_template(self.state, action, reward, done, state_)
        self.exp_buffer.append(exp)

        # 如果游戏结束，则返回累计奖励
        if done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


if __name__ == '__main__':
    set_global_seeds(0)
    env = get_env(DEFAULT_ENV_NAME, 0, 'dqn')

    net = DQN(env.observation_space.shape, env.action_space.n)
    target_net = DQN(env.observation_space.shape, env.action_space.n).load_state_dict(net.state_dict())

    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_reward = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()

    best_m_rewards = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        # 返回的奖励是智能体累计奖励
        reward = agent.play_step(net, epsilon)
        if reward is not None:
            total_reward.append(reward)
            # 计算速度
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            # 计算最近一百次的平均奖励
            m_reward = np.mean(total_reward[-100:])
            print("{}:done {}games, rewards {}, eps {}, speed {} f/s"
                  .format(frame_idx, len(total_reward), m_reward, epsilon, speed))

            # 如果获得了更好的结果
            if best_m_rewards is None or best_m_rewards < m_reward:
                torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best_{}.dat".format(m_reward))
                if best_m_rewards is not None:
                    print("best reward updated {} -> {}".format(best_m_rewards, m_reward))
                best_m_rewards = m_reward
            # 如果获得奖励大于阈值
            if m_reward >= MEAN_REWARD_ROUND:
                print("solved in {} frames".format(frame_idx))

        if len(buffer) < REPLAY_START_SIZE:
            continue

        # 每隔 SYNC_TARGET_FRAMES 同步一次网络
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            target_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calculate_loss(batch, net, target_net)
        loss_t.backward()
        optimizer.step()
