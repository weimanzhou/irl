'''
Author: snowflake
Date: 2021-12-19 14:31:16
LastEditTime: 2021-12-19 15:09:09
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \irl\src\q-learning\main.py
'''

import numpy as np 
import gym
import random
import time 
from IPython.display import clear_output


rewards_all_episode = []

num_episodes = 10000
max_steps_per_episode = 100

exploration_rate = 1
min_exploration_rate = 0.01
max_exploration_rate = 1
exploration_decay_rate = 0.01

learning_rate = 0.1
discount_rate = 0.99

env = gym.make('FrozenLake-v1')

state_space_size = env.observation_space.shape or env.observation_space.n 
action_space_size = env.action_space.shape or env.action_space.n

state_space_size = np.prod(state_space_size)
action_space_size = np.prod(action_space_size)

q_table = np.zeros((state_space_size, action_space_size))

# Q-Learning algorithm
for episode in range(num_episodes):
    # initialize new episode params
    state = env.reset()
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):

        # Exploration-exploitation trade-off 
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

        # Take new action 
        new_state, reward, done, info = env.step(action)

        # Update Q table 
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        # Set new state 
        state = new_state
        rewards_current_episode += reward

        # Add new reward
        if done == True:
            break

    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    rewards_all_episode.append(rewards_current_episode)

rewards_per_thousand_episodes = np.split(np.array(rewards_all_episode), num_episodes / 1000)
count = 1000 

for r in rewards_per_thousand_episodes:
    print("{} : {}".format(count, str(sum(r / 1000))))
    count += 1000

np.savetxt('src/q-learning/result.txt', q_table)