import numpy as np

from pettingzoo.mpe import simple_spread_v2


if __name__ == '__main__':
    env = simple_spread_v2.env()

    env.reset()
    """
        即使是多智能体环境，step每次调用仍然只能执行一次
        环境中内部会维护一个 agent_selection 对象，用来指向当前行动的智能体
    """
    env.step(1)

