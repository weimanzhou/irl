import sys
sys.path.extend(['C:\\Users\\snowflake\\Documents\\GITHUB\\irl', 'C:/Users/snowflake/Documents/GITHUB/irl'])
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
# from torch.utils.tensorboard import SummaryWriter

from src.env.MAGridEnvWrapper import GridEnv

# writer = SummaryWriter()


class BehaviorPolicyNN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(BehaviorPolicyNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class BehaviorCriticNN(nn.Module):
    def __init__(self, input_dim):
        super(BehaviorCriticNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


class EvaluationValueNN(nn.Module):
    def __init__(self, input_dim):
        super(EvaluationValueNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),	
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


class BehaviorModule(nn.Module):
    """
    :param input_dim 行为模块的策略网络
    :param output_dim 行为模块的状态值网络
    """

    def __init__(self, input_dim, output_dim):
        super(BehaviorModule, self).__init__()
        self.p_n = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.b_n = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.b_n_ = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.b_n_.load_state_dict(self.b_n.state_dict())
        self.gamma1 = 0.9

        self.distribution = torch.distributions.Categorical

    def choose_action(self, s):
        self.p_n.eval()
        self.b_n.eval()
        self.b_n_.eval()

        logits = self.p_n.forward(s)
        # prob = f.softmax(logits, dim=0).data
        # m = self.distribution(prob)
        # TODO 待检查
        action = torch.argmax(logits)
        return action
        # return m.sample().numpy()

    def loss(self, s, a, r, s_):
        self.p_n.train()
        self.b_n.train()
        self.b_n_.train()

        if s.dim() == 1:
            s = torch.unsqueeze(s, dim=0)
        if s_.dim() == 1:
            s_ = torch.unsqueeze(s_, dim=0)

        logits = self.p_n(s)
        values = self.b_n(s)

        td = self.gamma1 * self.b_n_(s_) + r - values
        c_loss = 0.5 * td.pow(2)

        # logits, values = self.forward(s)
        # td = v_t - values
        # S_loss = td.pow(2)
        #
        # probs = F.softmax(logits, dim=1)
        # m = self.distribution(probs)
        # exp_v = m.log_prob(a) * td.detach().squeeze()
        # P_loss = -exp_v
        probs = f.softmax(logits, dim=0)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        # total_loss = (c_loss + a_loss).mean()

        return c_loss, a_loss

    def get_params(self):
        return {
            "p_n": self.p_n.state_dict(),
            "b_n": self.b_n.state_dict(),
        }

    def set_params(self, params):
        self.p_n.load_state_dict(params['p_n'])
        self.b_n.load_state_dict(params['b_n'])

    def grad(self):
        return [p.grad for p in self.p_n.parameters()], [p.grad for p in self.b_n.parameters()]

    def set_grad(self, grad_avg):
        p_n_grad, b_n_grad = grad_avg
        for g1, g2, g1_, g2_ in zip(self.p_n.parameters(), self.b_n.parameters(), p_n_grad, b_n_grad):
            g1._grad = g1_
            g2._grad = g2_

    def update_target_network(self):
        self.b_n_.load_state_dict(self.b_n.state_dict())


class EvaluationModule(nn.Module):
    """
    :param input_dim 状态空间的大小
    """

    def __init__(self, input_dim):
        super(EvaluationModule, self).__init__()
        self.e_n = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.e_n_ = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.e_n_.load_state_dict(self.e_n.state_dict())

    def calculate_action_priority(self, s):
        if s.dim() == 1:
            s = torch.unsqueeze(s, dim=0)
        return self.e_n(s)

    def loss(self, s, s_, r):
        self.e_n.train()
        self.e_n_.train()

        values = self.e_n(s)
        values_ = self.e_n_(s_)

        c_loss = torch.pow(r + values_ - values, 2)

        return c_loss

    def get_params(self):
        return {
            "e_n": self.e_n.state_dict(),
        }

    def set_params(self, params):
        self.e_n.load_state_dict(params['e_n'])

    def grad(self):
        return [e.grad for e in self.e_n.parameters()]

    def set_grad(self, grad_avg):
        e_n_grad = grad_avg
        for e1, e1_ in zip(self.e_n.parameters(), e_n_grad):
            e1._grad = e1_

    def update_target_network(self):
        self.e_n_.load_state_dict(self.e_n.state_dict())


class Worker:
    """
    :param state_space_shape 状态空间形状
    :param action_space_shape 行为空间
    :param index work编号
    """

    def __init__(self, state_space_shape, action_space_shape, index):
        self.index = index

        state_n = np.prod(state_space_shape)
        action_n = np.prod(action_space_shape)

        # p_n = BehaviorPolicyNN(state_n, action_n)
        # b_n = BehaviorCriticNN(state_n)
        # b_n_ = BehaviorCriticNN(state_n)
        #
        # e_n = EvaluationValueNN(state_n)
        # e_n_ = EvaluationValueNN(state_n)

        self.behavior_module = BehaviorModule(state_n, action_n)
        self.evaluation_module = EvaluationModule(state_n)

    def calculate_action_priority(self, s):
        return self.evaluation_module.calculate_action_priority(s)

    def get_action(self, state):
        return self.behavior_module.choose_action(state)

    def calculate_grad_e(self, s, s_, r):
        loss = self.evaluation_module.loss(s, s_, r)
        # writer.add_scalar('evaluation_module/loss', loss.item())
        print("evaluation_module: loss:{:6f}".format(loss.item()))
        loss.backward()
        return self.evaluation_module.grad()

    def calculate_grad_b(self, s, a, s_, r):
        c_loss, a_loss = self.behavior_module.loss(s, a, r, s_)
        # writer.add_scalar('behavior_module/c_loss', c_loss.item())
        # writer.add_scalar('behavior_module/a_loss', a_loss.item())
        print("behavior_module: c_loss:{:6f}, a_loss:{:6f}".format(c_loss.item(), a_loss.item()))
        c_loss.backward()
        a_loss.backward()
        return self.behavior_module.grad()

    def get_params_b(self):
        return self.behavior_module.get_params()

    def get_params_e(self):
        return self.evaluation_module.get_params()

    def set_params_b(self, param):
        self.behavior_module.set_params(param)

    def set_params_e(self, param):
        self.evaluation_module.set_params(param)

    def set_grad_b(self, grad_avg):
        self.behavior_module.set_grad(grad_avg)

    def set_grad_e(self, grad_avg):
        self.evaluation_module.set_grad(grad_avg)

    def update_target_network(self):
        self.evaluation_module.update_target_network()
        self.behavior_module.update_target_network()


episodes = 10
tmax = 1000
lr = 1e-4
momentum = 0.8
target_update_freq = 10000
action_desc = ['停', '右', '左', '上', '下']


def train():
    # prof = profiler.profile(
    #     schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #     on_trace_ready=profiler.tensorboard_trace_handler('./log/isrl'),
    #     record_shapes=True,
    #     with_stack=True)
    # prof.start()
    env = GridEnv()
    states = env.reset()
    states = torch.tensor(states, dtype=torch.float32)
    # 获取环境中智能体的个数
    agent_count = env.agent_count
    state_space_shape = env.observation_space.shape
    action_space_shape = env.action_space.shape

    # 初始化智能体
    workers = [Worker(state_space_shape, action_space_shape, i) for i in range(agent_count)]
    virtual_worker = Worker(state_space_shape, action_space_shape, -1)

    # 初始化评估模块和行为模块的优化器
    optim_behavior = torch.optim.SGD(virtual_worker.behavior_module.parameters(), lr=lr, momentum=momentum)
    optim_evaluation = torch.optim.SGD(virtual_worker.evaluation_module.parameters(), lr=lr, momentum=momentum)
    count = 0
    for _ in range(episodes):
        t = 1
        global_reward = 0
        while t <= tmax:
            count += 1
            print("count: {}".format(count))
            if count % target_update_freq == 0:
                for worker in workers:
                    worker.update_targep_network()
            # evaluation module
            # 收集每一个智能体下一个状态和奖励
            states_, rewards = [], []
            for idx, worker in enumerate(workers):
                state = states[idx]
                action = worker.get_action(state)
                print('evaluation module agent: {} do action: {}'.format(idx, action_desc[action]))
                state_, reward, done, info = env.step_agent(idx, action)
                # print("info: {}".format(info))
                states_.append(state_)
                rewards.append(reward)

            if global_reward > 0:
                break

            # 收集每一个智能体的梯度
            grad_all = []
            for idx in range(agent_count):
                state = states[idx]
                state_ = states_[idx]
                reward = rewards[idx]

                state_ = torch.tensor(state_, dtype=torch.float32)
                reward = torch.tensor(reward, dtype=torch.float32)
                begin = time.time_ns()
                grad_e = workers[idx].calculate_grad_e(state, state_, reward)
                # print("evaluation_module consume time: {}".format(time.time_ns() - begin))
                grad_all.append(grad_e)

            # 对收集后的梯度进行加和平均，更新虚拟智能体的参数
            optim_evaluation.zero_grad()
            grad_all_numpy = [[_.numpy() for _ in g] for g in grad_all]
            grad_avg = np.sum(grad_all_numpy, axis=0) / len(grad_all_numpy)
            grad_avg = [torch.tensor(avg, dtype=torch.float32) for avg in grad_avg]
            virtual_worker.set_grad_e(grad_avg)
            optim_evaluation.step()

            # 每一个智能体更新参数
            for idx, worker in enumerate(workers):
                worker.set_params_e(virtual_worker.get_params_e())

            # behavior module
            action_priority_all = []
            for idx, worker in enumerate(workers):
                state = states[idx]
                action_priority = worker.calculate_action_priority(state)
                # action_priority = random.random()
                action_priority_all.append((action_priority, idx, state))

            # 这里已经获取了每一个智能体的优先级
            # 根据 coordination channel 的范围计算每一个智能体是否能够行动
            # dt = np.dtype([('action_priority', np.float32), ('idx', np.uint8), ('state', np.ndarray)])
            # can_action_agent = []
            # for action_priority, idx, state in action_priority_all:
            #     agents_neighbors = env.get_neighborhood(idx)
            #     agents_neighbors.append(idx)
            #     # TODO 对于一个智能体，获取到它周围优先级最大的那个智能体
            #     max_idx = idx
            #     for neighbor in agents_neighbors:
            #         if action_priority_all[neighbor] > action_priority_all[max_idx]:
            #             max_idx = neighbor
            #     # tmp = np.max(agents_neighbors)
            #     if max_idx == idx:
            #         can_action_agent.append((action_priority, idx, state))

            dt = np.dtype([('action_priority', np.float32), ('idx', np.uint8), ('state', np.ndarray)])
            action_priority_all = np.array(action_priority_all, dtype=dt)
            action_priority_all = np.sort(action_priority_all, order='action_priority')
            # 目前是假设所有的智能体都能够感知到其它全部智能体
            can_action_agent = action_priority_all[0:len(action_priority_all) // 2]

            if global_reward > 0:
                break

            # 更新每一个智能体的梯度
            grad_all = []
            for action_priority, idx, state in can_action_agent:
                states[idx] = state
                action = workers[idx].get_action(state)
                print('behavior module agent: {} do action: {}'.format(idx, action_desc[action]))
                state_, reward, done, info = env.step_agent(idx, action)
                # print("info: {}".format(info))
                state_ = torch.tensor(state_, dtype=torch.float32)
                # action = torch.tensor(action, dtype=torch.float32)
                begin = time.time_ns()
                grad_p, grad_b = workers[idx].calculate_grad_b(states[idx], action, state_, reward)
                # print("behavior_module consume time: {}".format(time.time_ns() - begin))
                grad_all.append((grad_p, grad_b))

            # 对收集后的梯度进行加和平均，更新虚拟智能体的参数
            optim_behavior.zero_grad()
            grad_all_numpy = [[[__.numpy() for __ in _] for _ in g] for g in grad_all]
            grad_avg = np.sum(grad_all_numpy, axis=0) / len(grad_all_numpy)
            grad_avg = [[torch.tensor(_, dtype=torch.float32) for _ in avg] for avg in grad_avg]
            # grad_avg = np.sum(grad_all, axis=0) / len(grad_all)
            virtual_worker.set_grad_b(grad_avg)
            optim_behavior.step()

            # 每一个智能体更新参数
            for idx, worker in enumerate(workers):
                worker.set_params_b(virtual_worker.get_params_b())

    #         prof.step()
    # prof.stop()


if __name__ == '__main__':
    train()
