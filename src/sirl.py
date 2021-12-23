'''
Author: your name
Date: 2021-12-22 16:58:30
LastEditTime: 2021-12-22 21:35:05
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \irl\src\sirl.py
'''

# 定义数据的轮数
episode = 10
tmax = 1000
env = MLGridEnv()


def train_evaluation_module():
    # train evaluation module 

    sample_random = sample_random()
    agents = init_agent_from_sample()

    for t in range(tmax):
        current_state = sample_random()
        for agent in agents:
            # 1. select an attractor according to (1) from the input local state 
            current_attractor = random.sample(current_state)

            # 2. send the local state to the Evaluation Module 
            evaluation_module(current_state)

            # 3. send the local state to Behavior Module
            action = behavior_module(current_state)

            # 4. perform the action 
            new_state, reward, done, info = env.step(action)

            # 5. modify the digital pheromone at current position according to (2)
            current_location = new_state.location
            if before_have(current_location):
                # diffuse
                diffuse(current_location)
                # superpose the amound of digital pheromone at the same position linearly
                superpose(current_location)

            # 6. calculate the individual reward according to (9)
            calculate_reward()

        # decay the amount of digital pheromone at position already occupied by agents with fixed decay rate

        if calculate_grobal_reward > 0:
            break
        else:
            for agent in agents:
                # calculate the grad of the state value network within Evaluation Module according to (4)-(5)
                grad = calculate_grad(evaluation_module)

                # send grad to v agent
                send_to_v_agent(grad)
            # v agent optim state value network within Evaluation Module according to (10)
            optim_grad(evaluation_module)

            # v agent send new theta to all agents
            new_theta = v_agent.get_new_theta()

            # all agents update theta_i to theta_{i+1} by new theta
            for agent in agents:
                agent.set_theta = new_theta

            


    pass


def train_behavior_module():

    # train behavior module
    sample_random = sample_random()
    agents = init_agent_from_sample()

    for t in range(tmax):
        current_state = sample_random()
        for agent in agents:
            # 1. select an attractor according to (1) from the input local state 
            current_attractor = random.sample(current_state)

            # 2. send the local state to the Evaluation Module and calculate the action priority
            actions = evaluation_module(current_state)

            # 3. send the local state to coordination and receive the return priority list
            
            # 4. perform the action 
            new_state, reward, done, info = env.step(action)

            # 5. modify the digital pheromone at current position according to (2)
            current_location = new_state.location
            if before_have(current_location):
                # diffuse
                diffuse(current_location)
                # superpose the amound of digital pheromone at the same position linearly
                superpose(current_location)

            # 6. calculate the individual reward according to (9)
            calculate_reward()

        # decay the amount of digital pheromone at position already occupied by agents with fixed decay rate

        if calculate_grobal_reward > 0:
            break
        else:
            for agent in agents:
                # calculate the grad of the state value network within Evaluation Module according to (4)-(5)
                grad = calculate_grad(evaluation_module)

                # send grad to v agent
                send_to_v_agent(grad)
            # v agent optim state value network within Evaluation Module according to (10)
            optim_grad(evaluation_module)

            # v agent send new theta to all agents
            new_theta = v_agent.get_new_theta()

            # all agents update theta_i to theta_{i+1} by new theta
            for agent in agents:
                agent.set_theta = new_theta

    pass

for round in range(episode):

    train_evaluation_module()
    train_behavior_module()

