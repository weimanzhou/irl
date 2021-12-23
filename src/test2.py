'''
Author: your name
Date: 2021-10-10 13:25:14
LastEditTime: 2021-10-10 14:41:03
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \imanim\test2.py
'''

from PyQt5 import QtCore
import numpy as np
import pandas as pd
import sys
import time
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import *
from QThread_Example_UI import Ui_Form

N_STATES = 16
ACTIONS = ['left', 'right', 'up', 'down']
EPSILON = 0.6
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 100
FRESH_TIME = 0.5

class Thread(QThread):
    trigger = pyqtSignal(int)

    def __init__(self) -> None:
        super(Thread, self).__init__()

    def run(self):
        self.rl()


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
            self.trigger.emit(S)

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


class MyMainForm(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        self.work = Thread()
        self.button.clicked.connect(self.execute)

    def execute(self):
        self.work.start()
        self.work.trigger.connect(self.display)

    def init_panel(self):
        for i in range(4):
            for j in range(4):
                frame = QLabel(str(i) + ':' + str(j))
                # self.frames.append(frame)
                frame.setFrameStyle(QFrame.Box)
                frame.setAlignment(QtCore.Qt.AlignCenter)
                self.grid.removeWidget(self.grid.itemAtPosition(i + 1, j + 1).widget())
                self.grid.addWidget(frame, i + 1, j + 1)

        self.grid.removeWidget(self.grid.itemAtPosition(3, 3).widget())
        self.grid.addWidget(QLabel('win'), 3, 3)

    def display(self, S):
        self.init_panel()
        row = S // 4 + 1
        col = S % 4 + 1
        
        widget = self.grid.itemAtPosition(row, col).widget()
        self.grid.removeWidget(widget)
        new = QLabel('here')
        new.setStyleSheet('background-color: red')
        self.grid.addWidget(new, row, col)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainForm()
    myWin.show()
    sys.exit(app.exec_())