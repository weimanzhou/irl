import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from PIL import ImageFont, ImageDraw, Image

a_1 = 1
b_1 = 0.9
diffusion_rate = 0.8


class EnvFindGoals(object):

    def __init__(self,
                 agent_count=4,
                 map_size=(30, 30)
                 ):

        self.agent_count = agent_count
        self.agent_position = np.random.randint(1, map_size[0], size=(agent_count, 2))
        self.rows = map_size[0]
        self.cols = map_size[1]

        maps = np.zeros((self.rows, self.cols))
        factor_maps = np.zeros_like(maps)
        target_maps = np.random.randint(0, 9, size=np.array(maps).shape)

        self.maps = np.pad(maps, ((1, 1), (1, 1)), mode='constant', constant_values=1)
        self.factor_maps = np.pad(factor_maps, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        self.target_maps = np.pad(target_maps, ((1, 1), (1, 1)), mode='constant', constant_values=0)

        self.occupancy = maps

    def list_add(self, a, b):
        c = [a[i] + b[i] for i in range(min(len(a), len(b)))]
        return c

    def get_agt1_obs(self):
        visual_range = 3
        vec = np.zeros((visual_range, visual_range, 3))

        for i in range(visual_range):
            for j in range(visual_range):
                vec[i, j, 0] = 1.0
                vec[i, j, 1] = 1.0
                vec[i, j, 2] = 1.0

        # detect block
        if self.occupancy[self.agt1_pos[0] - 1][self.agt1_pos[1] + 1] == 1:
            vec[0, 0, 0] = 0.0
            vec[0, 0, 1] = 0.0
            vec[0, 0, 2] = 0.0
        if self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] + 1] == 1:
            vec[0, 1, 0] = 0.0
            vec[0, 1, 1] = 0.0
            vec[0, 1, 2] = 0.0
        if self.occupancy[self.agt1_pos[0] + 1][self.agt1_pos[1] + 1] == 1:
            vec[0, 2, 0] = 0.0
            vec[0, 2, 1] = 0.0
            vec[0, 2, 2] = 0.0
        if self.occupancy[self.agt1_pos[0] - 1][self.agt1_pos[1]] == 1:
            vec[1, 0, 0] = 0.0
            vec[1, 0, 1] = 0.0
            vec[1, 0, 2] = 0.0
        if self.occupancy[self.agt1_pos[0] + 1][self.agt1_pos[1]] == 1:
            vec[1, 2, 0] = 0.0
            vec[1, 2, 1] = 0.0
            vec[1, 2, 2] = 0.0
        if self.occupancy[self.agt1_pos[0] - 1][self.agt1_pos[1] - 1] == 1:
            vec[2, 0, 0] = 0.0
            vec[2, 0, 1] = 0.0
            vec[2, 0, 2] = 0.0
        if self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] - 1] == 1:
            vec[2, 1, 0] = 0.0
            vec[2, 1, 1] = 0.0
            vec[2, 1, 2] = 0.0
        if self.occupancy[self.agt1_pos[0] + 1][self.agt1_pos[1] - 1] == 1:
            vec[2, 2, 0] = 0.0
            vec[2, 2, 1] = 0.0
            vec[2, 2, 2] = 0.0

        # detect self
        vec[1, 1, 0] = 1.0
        vec[1, 1, 1] = 0.0
        vec[1, 1, 2] = 0.0

        # detect agent2
        if self.agt2_pos == self.list_add(self.agt1_pos, [-1, 1]):
            vec[0, 0, 0] = 0.0
            vec[0, 0, 1] = 0.0
            vec[0, 0, 2] = 1.0
        if self.agt2_pos == self.list_add(self.agt1_pos, [0, 1]):
            vec[0, 1, 0] = 0.0
            vec[0, 1, 1] = 0.0
            vec[0, 1, 2] = 1.0
        if self.agt2_pos == self.list_add(self.agt1_pos, [1, 1]):
            vec[0, 2, 0] = 0.0
            vec[0, 2, 1] = 0.0
            vec[0, 2, 2] = 1.0
        if self.agt2_pos == self.list_add(self.agt1_pos, [-1, 0]):
            vec[1, 0, 0] = 0.0
            vec[1, 0, 1] = 0.0
            vec[1, 0, 2] = 1.0
        if self.agt2_pos == self.list_add(self.agt1_pos, [1, 0]):
            vec[1, 2, 0] = 0.0
            vec[1, 2, 1] = 0.0
            vec[1, 2, 2] = 1.0
        if self.agt2_pos == self.list_add(self.agt1_pos, [-1, -1]):
            vec[2, 0, 0] = 0.0
            vec[2, 0, 1] = 0.0
            vec[2, 0, 2] = 1.0
        if self.agt2_pos == self.list_add(self.agt1_pos, [0, -1]):
            vec[2, 1, 0] = 0.0
            vec[2, 1, 1] = 0.0
            vec[2, 1, 2] = 1.0
        if self.agt2_pos == self.list_add(self.agt1_pos, [1, -1]):
            vec[2, 2, 0] = 0.0
            vec[2, 2, 1] = 0.0
            vec[2, 2, 2] = 1.0
        return vec

    def get_agt2_obs(self):
        visual_range = 3
        vec = np.zeros((visual_range, visual_range, 3))

        for i in range(visual_range):
            for j in range(visual_range):
                vec[i, j, 0] = 1.0
                vec[i, j, 1] = 1.0
                vec[i, j, 2] = 1.0

        # detect block
        if self.occupancy[self.agt2_pos[0] - 1][self.agt2_pos[1] + 1] == 1:
            vec[0, 0, 0] = 0.0
            vec[0, 0, 1] = 0.0
            vec[0, 0, 2] = 0.0
        if self.occupancy[self.agt2_pos[0]][self.agt2_pos[1] + 1] == 1:
            vec[0, 1, 0] = 0.0
            vec[0, 1, 1] = 0.0
            vec[0, 1, 2] = 0.0
        if self.occupancy[self.agt2_pos[0] + 1][self.agt2_pos[1] + 1] == 1:
            vec[0, 2, 0] = 0.0
            vec[0, 2, 1] = 0.0
            vec[0, 2, 2] = 0.0
        if self.occupancy[self.agt2_pos[0] - 1][self.agt2_pos[1]] == 1:
            vec[1, 0, 0] = 0.0
            vec[1, 0, 1] = 0.0
            vec[1, 0, 2] = 0.0
        if self.occupancy[self.agt2_pos[0] + 1][self.agt2_pos[1]] == 1:
            vec[1, 2, 0] = 0.0
            vec[1, 2, 1] = 0.0
            vec[1, 2, 2] = 0.0
        if self.occupancy[self.agt2_pos[0] - 1][self.agt2_pos[1] - 1] == 1:
            vec[2, 0, 0] = 0.0
            vec[2, 0, 1] = 0.0
            vec[2, 0, 2] = 0.0
        if self.occupancy[self.agt2_pos[0]][self.agt2_pos[1] - 1] == 1:
            vec[2, 1, 0] = 0.0
            vec[2, 1, 1] = 0.0
            vec[2, 1, 2] = 0.0
        if self.occupancy[self.agt2_pos[0] + 1][self.agt2_pos[1] - 1] == 1:
            vec[2, 2, 0] = 0.0
            vec[2, 2, 1] = 0.0
            vec[2, 2, 2] = 0.0

        # detect self
        vec[1, 1, 0] = 0.0
        vec[1, 1, 1] = 0.0
        vec[1, 1, 2] = 1.0

        # detect agent2
        if self.agt1_pos == self.list_add(self.agt2_pos, [-1, 1]):
            vec[0, 0, 0] = 1.0
            vec[0, 0, 1] = 0.0
            vec[0, 0, 2] = 0.0
        if self.agt1_pos == self.list_add(self.agt2_pos, [0, 1]):
            vec[0, 1, 0] = 1.0
            vec[0, 1, 1] = 0.0
            vec[0, 1, 2] = 0.0
        if self.agt1_pos == self.list_add(self.agt2_pos, [1, 1]):
            vec[0, 2, 0] = 1.0
            vec[0, 2, 1] = 0.0
            vec[0, 2, 2] = 0.0
        if self.agt1_pos == self.list_add(self.agt2_pos, [-1, 0]):
            vec[1, 0, 0] = 1.0
            vec[1, 0, 1] = 0.0
            vec[1, 0, 2] = 0.0
        if self.agt1_pos == self.list_add(self.agt2_pos, [1, 0]):
            vec[1, 2, 0] = 1.0
            vec[1, 2, 1] = 0.0
            vec[1, 2, 2] = 0.0
        if self.agt1_pos == self.list_add(self.agt2_pos, [-1, -1]):
            vec[2, 0, 0] = 1.0
            vec[2, 0, 1] = 0.0
            vec[2, 0, 2] = 0.0
        if self.agt1_pos == self.list_add(self.agt2_pos, [0, -1]):
            vec[2, 1, 0] = 1.0
            vec[2, 1, 1] = 0.0
            vec[2, 1, 2] = 0.0
        if self.agt1_pos == self.list_add(self.agt2_pos, [1, -1]):
            vec[2, 2, 0] = 1.0
            vec[2, 2, 1] = 0.0
            vec[2, 2, 2] = 0.0
        return vec

    def get_full_obs(self):
        obs = np.ones((8, 10, 3))
        for i in range(8):
            for j in range(10):
                if self.occupancy[j][i] == 1:
                    obs[3 - i, j, 0] = 0
                    obs[3 - i, j, 1] = 0
                    obs[3 - i, j, 2] = 0
                if [j, i] == self.agt1_pos:
                    obs[3 - i, j, 0] = 1
                    obs[3 - i, j, 1] = 0
                    obs[3 - i, j, 2] = 0
                if [j, i] == self.agt2_pos:
                    obs[3 - i, j, 0] = 0
                    obs[3 - i, j, 1] = 0
                    obs[3 - i, j, 2] = 1
        return obs

    def get_obs(self):
        return [self.get_agt1_obs(), self.get_agt2_obs()]

    def reset(self):
        self.agent_position = np.random.randint(1, self.cols, size=(self.agent_count, 2))
        self.agent_current = 0

        self.panel_x = (self.cols + 2) * 20
        self.panel_y = 0

    def step_agent(self, agent, action):
        rewards = np.zeros(self.agent_count)

        can_actions = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        action = can_actions[action]
        agent = self.agent_position[agent]
        if self.maps[agent[0] + action[0]][agent[1] + action[1]] == 0:
            self.maps[agent[0]][agent[1]] = 0
            agent[0] = agent[0] + action[0]
            agent[1] = agent[1] + action[1]
            self.maps[agent[0]][agent[1]] = 1
        else:
            rewards[agent] -= 3

        if self.target_maps[agent[0]][agent[1]]:
            self.maps[agent[0]][agent[1]] = 0
            agent = self.agent_position[i]
            self.maps[agent[0]][agent[1]] = 1
            rewards[i] += 50

        done = False
        if rewards[0] > 0:
            done = True

        return rewards, done

    def step(self, action_list):
        rewards = np.zeros(self.agent_count)

        can_actions = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        for i in range(len(action_list)):
            # print(action_list)
            action = can_actions[action_list[i]]
            agent = self.agent_position[i]
            if self.maps[agent[0] + action[0]][agent[1] + action[1]] == 0:
                self.maps[agent[0]][agent[1]] = 0
                agent[0] = agent[0] + action[0]
                agent[1] = agent[1] + action[1]
                self.maps[agent[0]][agent[1]] = 1
            else:
                rewards[i] -= 3

            if self.target_maps[agent[0]][agent[1]]:
                self.maps[agent[0]][agent[1]] = 0
                agent = self.agent_position[i]
                self.maps[agent[0]][agent[1]] = 1
                rewards[i] += 50

        self._modify_digital_pheromone()
        self._diffuse_digital_pheromone()
        self._superpose_digital_pheromone()

        done = False
        if rewards[0] > 0:
            done = True
        return rewards, done

    def plot_scene(self):
        fig = plt.figure(figsize=(5, 5))
        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        plt.xticks([])
        plt.yticks([])
        ax2 = fig.add_subplot(gs[2, 0:1])
        plt.xticks([])
        plt.yticks([])
        ax3 = fig.add_subplot(gs[2, 1:2])
        plt.xticks([])
        plt.yticks([])
        ax1.imshow(self.get_full_obs())
        ax2.imshow(self.get_agt1_obs())
        ax3.imshow(self.get_agt2_obs())

        plt.show()

    def _draw_text(self, img, text, position):
        font = ImageFont.truetype("msyhl.ttc", 15)
        obs_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(obs_pil)
        draw.text(position, text, font=font, fill=(0, 0, 0), align='center')
        obs = np.array(obs_pil)
        return obs

    def render(self):
        obs = np.full(((self.rows + 2) * 20, (self.cols + 2 + self.cols // 2) * 20, 3), 255, np.uint8)

        for i in range(self.rows + 2):
            for j in range(self.cols + 2):
                if self.maps[i][j] == 1:
                    # cv2.rectangle(obs, (j*20, i*20), (j*20+20, i*20+20), (0, 0, 0), -1)
                    image_wall = cv2.imread('./imgs/wall.png', 1)
                    obs[i * 20:i * 20 + 20, j * 20:j * 20 + 20] = image_wall
                if self.factor_maps[i][j] != 0:
                    # print(int(self.factor_maps[i][j] * 100) % 255)
                    cv2.rectangle(obs,
                                  (j * 20, i * 20),
                                  (j * 20 + 20, i * 20 + 20),
                                  (int(self.factor_maps[i][j] * 100 % 255), 255, 255),
                                  -1)
                    obs = self._draw_text(obs, str(int(self.factor_maps[i][j] * 10)), (j * 20, i * 20))
                if self.target_maps[i][j] == 1:
                    image_target = cv2.imread('./imgs/2.png', 1)
                    obs[i * 20:i * 20 + 20, j * 20:j * 20 + 20] = image_target
        for i in range(self.agent_count):
            # cv2.rectangle(obs,
            #               (self.agent_position[i][0] * 20, (7-self.agent_position[i][1]) * 20),
            #               (self.agent_position[i][0] * 20 + 20, (7-self.agent_position[i][1]) * 20 + 20),
            #               (255, 0, 0)
            #               -1)
            image = cv2.imread('./imgs/1.png', 1)
            agent_position = self.agent_position[i]
            obs[agent_position[0] * 20:agent_position[0] * 20 + 20,
                agent_position[1] * 20:agent_position[1] * 20 + 20] = image

        obs = self._draw_text(obs, "分值面板", (self.panel_x, self.panel_y))
        cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        cv2.imshow('image', obs)
        cv2.waitKey(300)

    '''
        动作执行之后，重新计算信息素的值
    '''

    def _modify_digital_pheromone(self):
        for agent in self.agent_position:
            row = agent[0]
            col = agent[1]
            if self.target_maps[row][col]:
                self.factor_maps[row][col] += a_1
            else:
                self.factor_maps[row][col] *= b_1

    '''
        衰减信息素的值
    '''

    def _diffuse_digital_pheromone(self):
        self.factor_maps *= diffusion_rate

    '''
        累加信息素的值
    '''

    def _superpose_digital_pheromone(self):
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

                self.factor_maps[row + move_x][col + move_y] += a_1


if __name__ == '__main__':

    agent_count = 10

    env = EnvFindGoals(agent_count=agent_count)
    env.reset()
    max_iter = 10000
    for i in range(max_iter):
        print("iter= ", i)
        action_list = np.random.randint(0, 4, agent_count)
        reward_list, done = env.step(action_list)
        env.render()
