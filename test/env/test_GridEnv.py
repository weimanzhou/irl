'''
Author: your name
Date: 2021-12-24 20:29:07
LastEditTime: 2021-12-24 21:34:25
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AEn
FilePath: \irl\src\env\test_GridEnv.py
'''
import sys, os 
# sys.path.insert(0, '../../')
sys.path.extend(['E:\\GITHUB\\irl', 'E:/GITHUB/irl'])
from src.env.MAGridEnvWrapper import GridEnv

import tianshou as ts


env = GridEnv()

observation_shape = env.observation_space
action_shape = env.action_space

print(observation_shape)
print(action_shape)