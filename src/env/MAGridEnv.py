import sys
sys.path.extend(['C:\\Users\\snowflake\\Documents\\GITHUB\\irl', 'C:/Users/snowflake/Documents/GITHUB/irl'])
import random

import numpy as np
import cv2
from PIL import ImageDraw, Image

from src.env.imgs import img


class EnvFindGoals(object):
    """
    :param agent_count count of agent
    :param map_size size of map
    :param visual_field
    :param a1
    :param a2
    :param a3
    :param b1
    :param b2
    :param b3
    :param c1
    :param diffusion_rate
    :param channel_range
    """

    def __init__(self,
                 agent_count=4,
                 target_count=4,
                 map_size=(30, 30),
                 visual_field=6,
                 a1=1,
                 a2=1,
                 a3=1,
                 b1=0.9,
                 b2=0,
                 b3=0.9,
                 c1=0.25,
                 p1=0.8,
                 diffusion_rate=0.8,
                 channel_range=4
                 ):

        self.agent_current = None
        self.agent_count = agent_count
        self.target_count = target_count
        self.agent_position = np.random.randint(1, map_size[0], size=(self.agent_count, 2))
        self.visual_field = visual_field
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.c1 = c1
        self.p1 = p1
        self.diffusion_rate = diffusion_rate
        # self.target_shape = target_shape
        self.channel_range = channel_range
        self.rows = map_size[0]
        self.cols = map_size[1]

        maps = np.zeros((self.rows, self.cols))
        factor_maps = np.full_like(maps, -1)
        target_maps = self._generate_digit_img(self.target_count, self.rows)

        # target_maps = np.where(target_maps == 1, 1, 0)

        self.maps = np.pad(maps, ((1, 1), (1, 1)), mode='constant', constant_values=1)
        self.factor_maps = np.pad(factor_maps, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        self.target_maps = np.pad(target_maps, ((1, 1), (1, 1)), mode='constant', constant_values=0)

        self.occupancy = maps

    # 获取全部智能体的状态
    def _state(self):
        state_all = []
        for agent in range(self.agent_count):
            state_all.append(self._state_by_agent(agent))

        return np.array(state_all)

    def _can_move_position(self, agent_current_position, move_x, move_y):
        """
            判断指定的智能体的在 x 轴方向能否移动 move_x，在 y 轴方向能够移动 move_y
        """
        return 0 < agent_current_position[1] + move_x < self.cols + 2 \
               and 0 < agent_current_position[0] + move_y < self.rows + 2

    def _can_move(self, agent, move_x, move_y):
        """
            判断指定的智能体的在 x 轴方向能否移动 move_x，在 y 轴方向能够移动 move_y
        """
        agent_current_position = self.agent_position[agent]

        return 0 < agent_current_position[1] + move_x < self.cols + 2 \
               and 0 < agent_current_position[0] + move_y < self.rows + 2

    def _calculate_d(self, agent_position, attractor_position):
        """
            计算 agent 和 attractor 之间的曼哈顿距离
        """
        distance = np.sum(np.abs(agent_position - attractor_position))

        return self.a2 * np.exp(-np.square(distance - self.b2) / 2 * np.square(self.c1))

    def _attractor_by_agent_position(self, agent_current_position):
        """
            计算指定智能体的 attractor
        """
        attractor_value = 0
        attractor_position = np.array([-1, -1])

        # 循环迭代视野范围，寻找最大的 attractor
        for i in np.arange(-self.visual_field, self.visual_field + 1, 1):
            for j in np.arange(-self.visual_field, self.visual_field + 1, 1):
                if np.abs(i) + np.abs(j) <= self.visual_field:
                    move_x = i
                    move_y = j

                    is_can_move = self._can_move_position(agent_current_position, move_x, move_y)

                    d = self._calculate_d(agent_current_position, np.array(
                        [agent_current_position[1] + move_x, agent_current_position[0] + move_y]))

                    if is_can_move and d * self.factor_maps[
                        agent_current_position[1] + move_x, agent_current_position[0] + move_y] > attractor_value:
                        attractor_value = self.factor_maps[
                            agent_current_position[1] + move_x, agent_current_position[0] + move_y]
                        attractor_position = np.array(
                            [agent_current_position[1] + move_x, agent_current_position[0] + move_y])

        return attractor_value, attractor_position

    def _attractor_by_agent(self, agent):
        """
            计算指定智能体的 attractor
        """
        # 获取当前智能体的位置
        agent_current_position = self.agent_position[agent]

        attractor_value = 0
        attractor_position = np.array([-1, -1])

        # 循环迭代视野范围，寻找最大的 attractor
        for i in np.arange(-self.visual_field, self.visual_field + 1, 1):
            for j in np.arange(-self.visual_field, self.visual_field + 1, 1):
                if np.abs(i) + np.abs(j) <= self.visual_field:
                    move_x = i
                    move_y = j

                    is_can_move = self._can_move(agent, move_x, move_y)

                    d = self._calculate_d(agent_current_position, np.array(
                        [agent_current_position[1] + move_x, agent_current_position[0] + move_y]))

                    if is_can_move and d * self.factor_maps[
                        agent_current_position[1] + move_x, agent_current_position[0] + move_y] > attractor_value:
                        attractor_value = self.factor_maps[
                            agent_current_position[1] + move_x, agent_current_position[0] + move_y]
                        attractor_position = np.array(
                            [agent_current_position[1] + move_x, agent_current_position[0] + move_y])

        return attractor_value, attractor_position

    def _state_by_agent(self, agent):
        """
            获取指定智能体的状态
        """
        # 获取当前智能体的位置
        agent_current_position = self.agent_position[agent]

        # 四个方向是否有智能体
        is_u_ok = self.maps[agent_current_position[0] - 1][agent_current_position[1]] != 1 \
            if agent_current_position[0] - 1 > 0 else False
        is_r_ok = self.maps[agent_current_position[0]][agent_current_position[1] + 1] != 1 \
            if agent_current_position[1] + 1 < self.cols + 2 else False
        is_d_ok = self.maps[agent_current_position[0] + 1][agent_current_position[1]] != 1 \
            if agent_current_position[0] + 1 < self.rows + 2 else False
        is_l_ok = self.maps[agent_current_position[0]][agent_current_position[1] - 1] != 1 \
            if agent_current_position[1] - 1 > 0 else False

        # 首先计算 attractor 的位置，然后计算出相对于 attractor 的位置
        attractor_value, attractor = self._attractor_by_agent(agent)

        relative_x = agent_current_position[1] - attractor[1]
        relative_y = agent_current_position[0] - attractor[0]

        # 当前位置是否是目标位置
        is_h_target = self.target_maps[agent_current_position[0]][agent_current_position[1]] == 1

        return np.array([is_u_ok, is_r_ok, is_d_ok, is_l_ok, relative_x, relative_y, is_h_target])

    def reset(self) -> np.ndarray:
        maps = np.zeros((self.rows, self.cols))
        factor_maps = np.full_like(self.maps, -1)
        target_maps = self._generate_digit_img(self.target_count, self.rows)

        self.maps = np.pad(maps, ((1, 1), (1, 1)), mode='constant', constant_values=1)
        self.factor_maps = np.pad(factor_maps, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        self.target_maps = np.pad(target_maps, ((1, 1), (1, 1)), mode='constant', constant_values=0)

        self.agent_count = np.count_nonzero(target_maps)
        self.agent_position = np.random.randint(1, self.rows, size=(self.agent_count, 2))

        self.agent_current = random.randint(0, self.agent_count - 1)
        self.panel_x = (self.cols + 2) * 20
        self.panel_y = 0
        self.rewards = np.zeros(self.agent_count)

        return np.array([self._state_by_agent(i) for i in range(self.agent_count)])

        # return self._state_by_agent(self.agent_current)

    def step_agent(self, agent: np.uint8, action: np.uint8):
        actions = np.zeros(self.agent_count, dtype=np.uint8)
        actions[agent] = action

        states_, rewards, dones, info = self.step(actions)
        info['neighbors'] = self.get_neighborhood(agent)
        return states_[agent], rewards[agent], dones, info

    def step(self, action_list):
        """
        :param action_list 动作列表 停，右，左，上，下
        """
        can_actions = np.array([[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0]])
        for i in range(len(action_list)):
            # print(action_list)
            action = can_actions[action_list[i]]
            agent = self.agent_position[i]
            if self.maps[agent[0] + action[0]][agent[1] + action[1]] == 0:
                self.maps[agent[0]][agent[1]] = 0
                agent[0] = agent[0] + action[0]
                agent[1] = agent[1] + action[1]
                self.maps[agent[0]][agent[1]] = 1

            # 设置奖励
            position_pre_x = agent[0] - action[0]
            position_pre_y = agent[1] - action[1]

            # if self.target_maps[agent[0]][agent[1]] == 0:
            #     self.rewards[i] = 0
            # if self.target_maps[position_pre_x][position_pre_y] and \
            #     self.target_maps[agent[0]][agent[1]]:
            #     # TODO 休要设置 detal SI
            #     self.rewards[i] = self.b3 * max(1, 1)
            # if self.target_maps[position_pre_x][position_pre_y] == 0 and \
            #     self.target_maps[agent[0]][agent[1]]:
            #     self.rewards[i] = self.a3

            # if self.target_maps[agent[0]][agent[1]]:
            #     self.maps[agent[0]][agent[1]] = 0
            #     agent = self.agent_position[i]
            #     self.maps[agent[0]][agent[1]] = 1
            #     self.rewards[i] += 1

        self._modify_digital_pheromone()
        self._diffuse_digital_pheromone()
        self._superpose_digital_pheromone()

        for i in range(len(action_list)):
            action = can_actions[action_list[i]]
            agent = self.agent_position[i]
            position_pre_x = agent[0] - action[0]
            position_pre_y = agent[1] - action[1]

            d1 = self._calculate_d(agent, self._attractor_by_agent_position((position_pre_x, position_pre_y))[1])
            d2 = self._calculate_d(agent, self._attractor_by_agent(i)[1])
            distance = d1 - d2

            self.rewards[i] = self.p1 * max(distance, 0)

        # self.agent_current = (self.agent_current + 1) % self.agent_count

        # self.factor_maps = np.vectorize(self._sigmoid)(self.target_maps)

        dones = np.empty(self.agent_count)
        dones_count = 0
        for index, agent in enumerate(self.agent_position):
            if self.target_maps[agent[0], agent[1]]:
                dones[index] = True
                dones_count += 1
        states_ = np.empty((self.agent_count, 7))
        for index in range(self.agent_count):
            states_[index, :] = self._state_by_agent(index)

        return states_, self.rewards, dones, \
               {"count_total": self.agent_count, "count_done": dones_count}

    def _draw_text(self, img, text, position):
        # font = ImageFont.truetype("msyhl.ttc", 15)
        obs_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(obs_pil)
        draw.text(position, text, fill=(0, 0, 0), align='center')
        obs = np.array(obs_pil)
        return obs

    def render(self, mode='human'):
        obs = np.full(((self.rows + 2) * 20, (self.cols + 2 + self.cols // 2 + self.cols + 2) * 20, 3), 255, np.uint8)

        target_offset_x = self.cols + 2 + self.cols // 2
        target_offset_y = 0

        for i in range(self.rows + 2):
            for j in range(self.cols + 2):
                if self.maps[i][j] == 1:
                    # cv2.rectangle(obs, (j*20, i*20), (j*20+20, i*20+20), (0, 0, 0), -1)
                    # image_wall = cv2.imread('src/env/imgs/wall.png', 1)
                    image_wall = np.where(img.wall != 0, img.wall, 255)
                    # obs[i * 20:i * 20 + 20, j * 20:j * 20 + 20] = image_wall
                    obs[i * 20:i * 20 + 20, j * 20:j * 20 + 20] = image_wall
                if self.factor_maps[i][j] > 1e-2:
                    # print(int(self.factor_maps[i][j] * 100) % 255)
                    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2HLS)
                    cv2.rectangle(obs,
                                  (j * 20, i * 20),
                                  (j * 20 + 20, i * 20 + 20),
                                  (238, 60, np.round(self.factor_maps[i][j] * 100)),
                                  -1)
                    obs = cv2.cvtColor(obs, cv2.COLOR_HLS2BGR)
                    # obs = self._draw_text(obs, str(np.round(self.factor_maps[i][j] * 100)), (j * 20, i * 20))
                if self.target_maps[i][j] == 1:
                    tmp = obs[i * 20:i * 20 + 20, j * 20:j * 20 + 20]

                    # image_target = cv2.imread('src/env/imgs/2.png', 1)
                    image_target = np.where(img.target != 0, img.target, tmp)

                    # tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2HLS)
                    # tmp = tmp & image_target
                    # tmp = cv2.cvtColor(tmp, cv2.COLOR_HLS2BGR)
                    # tmp = np.where(tmp != 0, tmp, 255)

                    obs[i * 20:i * 20 + 20, j * 20:j * 20 + 20] = image_target
        for i in range(self.rows + 2):
            for j in range(self.cols + 2):
                if self.maps[i][j] == 1:
                    # cv2.rectangle(obs, (j*20, i*20), (j*20+20, i*20+20), (0, 0, 0), -1)
                    # image_wall = cv2.imread('src/env/imgs/wall.png', 1)
                    image_wall = np.where(img.wall != 0, img.wall, 255)

                    obs[(i + target_offset_y) * 20:(i + target_offset_y) * 20 + 20, \
                    (j + target_offset_x) * 20:(j + target_offset_x) * 20 + 20] = image_wall
                if self.factor_maps[i][j] > 1e-2:
                    # print(np.round(self.factor_maps[i][j] * 255))
                    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2HLS)
                    cv2.rectangle(obs,
                                  ((j + target_offset_x) * 20, (i + target_offset_y) * 20),
                                  ((j + target_offset_x) * 20 + 20, (i + target_offset_y) * 20 + 20),
                                  (238, 60, np.round(self.factor_maps[i][j] * 100)),
                                  -1)
                    obs = cv2.cvtColor(obs, cv2.COLOR_HLS2BGR)
                    # obs = self._draw_text(obs, str(np.round(self.factor_maps[i][j] * 255)),
                    #                       ((j + target_offset_x) * 20, (i + target_offset_y) * 20))
        for i in range(self.agent_count):
            # cv2.rectangle(obs,
            #               (self.agent_position[i][0] * 20, (7-self.agent_position[i][1]) * 20),
            #               (self.agent_position[i][0] * 20 + 20, (7-self.agent_position[i][1]) * 20 + 20),
            #               (255, 0, 0)
            #               -1)
            agent_position = self.agent_position[i]
            # cv2.imread('src/env/imgs/1.png', 1)
            image = img.car
            tmp = obs[agent_position[0] * 20:agent_position[0] * 20 + 20,
                  agent_position[1] * 20:agent_position[1] * 20 + 20]
            image = np.where(image != 0, image, tmp)
            obs[agent_position[0] * 20:agent_position[0] * 20 + 20,
            agent_position[1] * 20:agent_position[1] * 20 + 20] = image

        obs = self._draw_text(obs, "score", (self.panel_x, self.panel_y))

        cv2.imshow('map', obs)
        cv2.waitKey(1)

    def close(self):
        pass

    def _generate_digit_img(self, num, n):
        img = np.zeros([n, n], np.uint8)
        targets = np.random.choice(self.rows * self.cols, self.target_count, replace=False)
        for idx in targets:
            img[idx // self.rows, idx % self.cols] = 1
        # cv2.putText(img, str(num), (0, n), cv2.FONT_HERSHEY_PLAIN, n // 10, (255, 255, 255))
        # 获取二值矩阵
        return img #np.where(img == 0, img, 1)[:, :, 0]

    def _sigmoid(self, x):
        """
            动作执行之后，重新计算信息素的值
        """
        return 1 / (1 + np.exp(-x))

    def _modify_digital_pheromone(self):
        for agent in self.agent_position:
            row = agent[0]
            col = agent[1]
            if self.target_maps[row][col]:
                self.factor_maps[row][col] += self.a1
            else:
                self.factor_maps[row][col] *= self.b1

    def _diffuse_digital_pheromone(self):
        """
            衰减信息素的值
        """
        self.factor_maps *= self.diffusion_rate

    def _superpose_digital_pheromone(self):
        """
            累加信息素的值
        """
        superpose = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        for i in range(self.agent_count):
            agent = self.agent_position[i]
            row = agent[0]
            col = agent[1]

            for j in range(len(superpose)):

                move = superpose[j]
                move_x = move[0]
                move_y = move[1]

                if row + move_x == 0 or row + move_x == self.rows + 1 \
                        or col + move_y == 0 or col + move_y == self.cols + 1:
                    continue

                self.factor_maps[row + move_x][col + move_y] += self.a1

    def get_neighborhood(self, idx):
        """
        获取一个智能体附近的邻居
        :param idx: 智能体索引
        :return:    智能体邻居的索引
        """
        agent_position = self.agent_position[idx]
        agent_neighborhoods = []
        for idx, agent_neighbor_position in enumerate(self.agent_position):
            distance = np.sum(np.abs(agent_position - agent_neighbor_position))
            if distance < self.channel_range:
                agent_neighborhoods.append(idx)
        return agent_neighborhoods


if __name__ == '__main__':

    agent_count_c = 4

    env = EnvFindGoals(agent_count=agent_count_c)
    env.reset()
    max_iter = 10000

    for i in range(5):
        state, reward_list, done, info = env.step_agent(0, i)
        print(env.agent_position[0])

    # while True:
    #     action_list_ = np.random.randint(0, 4, agent_count_c)
    #     state, reward_list, done, info = env.step(action_list_)
    #     env.render()
