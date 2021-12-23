'''
Author: your name
Date: 2021-10-08 21:26:04
LastEditTime: 2021-10-10 13:21:10
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \imanim\ttk.py
'''
import numpy as np
import pandas as pd
import time
from tkinter import *
from tkinter import ttk

import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *

N_STATES = 16
ACTIONS = ['left', 'right', 'up', 'down']
EPSILON = 0.6
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 100
FRESH_TIME = 0.01

class Test(QMainWindow):

    def __init__(self, parent=None) -> None:
        super(Test, self).__init__(parent)
        self.frames = []
        self.windows = QWidget()

        self.windows.resize(200, 200)
        self.windows.move(100, 100)

        grid = QGridLayout()
        self.windows.setLayout(grid)
        for i in range(4):
            for j in range(4):
                frame = QLabel(str(i) + ':' + str(j))
                # self.frames.append(frame)
                frame.setFrameStyle(QFrame.Box)
                grid.addWidget(frame, i + 1, j + 1)

        self.grid = grid

        self.button = QPushButton('start')
        self.grid.addWidget(self.button, 5, 5)
        self.button.clicked.connect(self.rl)

    def build_q_table(self, n_states, actions):
        self.table = pd.DataFrame(
            np.zeros((n_states, len(actions))),
            columns=actions
        )

    # 根据当前 state 从 q_table 中选取最适合的动作
    def choose_action(self, state, q_table):
        state_actions = q_table.iloc[state, :]
        if (np.random.uniform() > EPSILON or state_actions.all() == 0):
            action_name = np.random.choice(ACTIONS)
        else:
            action_name = state_actions.idxmax()
        return action_name


    # 根据动作获取下一步的状态和奖励
    def get_env_feedback(self, S, A):
        if A == 'right':
            if S == 9:
                S_ = 'terminal'
                R = 1
            else:
                S_ = S + 1 if S % 4 != 3 else S
                R = 0
        elif A == 'left':
            if S == 11:
                S_ = 'terminal'
                R = 1
            else:
                S_ = S - 1 if S % 4 != 0 else S
                R = 0
        elif A == 'up':
            if S == 14:
                S_ = 'terminal'
                R = 1
            else:
                S_ = S - 4 if S > 3 else S
                R = 0
        else:
            if S == 6:
                S_ = 'terminal'
                R = 1
            else:
                S_ = S + 4 if S < 12 else S
                R = 0
            
        return S_, R


    # S 当前状态
    # episode 轮数
    # step_couter 移动次数
    def update_env(self, S, episode, step_counter):
        if S == 'terminal':
            time.sleep(2)
        else:
            row = S // 4 + 1
            col = S % 4 + 1
            widget = self.grid.itemAtPosition(row, col).widget()
            self.grid.removeWidget(widget)
            self.grid.addWidget(QLabel('here'), row, col)
            # env_list[S] = 'o'
            # interaction = ''.join(env_list)
            # print('\r{}'.format(interaction), end='')
            time.sleep(FRESH_TIME)


    def rl(self):
        self.build_q_table(N_STATES, ACTIONS)
        q_table = self.table
        for episode in range(MAX_EPISODES):
            step_counter = 0
            S = np.random.randint(0, 16)
            while S == 10:
                S = np.random.randint(0, 16)
            is_terminated = False
            # 初始化环境
            self.update_env(S, episode, step_counter)
            while not is_terminated:
                A = self.choose_action(S, q_table)
                # 返回的动作和奖励
                S_, R = self.get_env_feedback(S, A)
                
                if S_ != 'terminal':
                    q_target = R + GAMMA * q_table.iloc[S_, :].max()
                else:
                    q_target = R
                    is_terminated = True
            
                q_predict = q_table.loc[S, A]
                q_table.loc[S, A] += ALPHA * (q_target - q_predict)
                S = S_ 

                step_counter += 1
                self.update_env(S, episode, step_counter)
                print("current: {}, count: {}", S, step_counter)

        return q_table

if __name__ == '__main__':
    app = QApplication(sys.argv)

    test = Test()
    test.show()
    # q_table = test.rl()
    print('\r\nQ-table:\n')
    sys.exit(app.exec_())
    # print(q_table)
