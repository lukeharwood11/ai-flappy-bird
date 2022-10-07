import os

import numpy as np
from pygame import K_UP, K_SPACE
import pygame
import utils
from position import Position
from constants import FLAP_SPEED, SCREEN_SIZE, THRUST, GRAVITY, GAME_HEIGHT, MAP_MOVEMENT, MODEL_INPUTS


class CollisionSet:

    def __init__(self, mini_batch_size):
        self.data = np.zeros(mini_batch_size)
        self._mini_batch_size = mini_batch_size

    def set_collision(self, index):
        self.data[index] = 1

    def crash_list(self):
        return self.data == 1

    def collision_at(self, index):
        return self.data[index] == 1

    def full_collision(self):
        return self.data == np.ones(len(self.data))

    def clear(self, mini_batch_size):
        """
        mini_batch_size can be smaller than self.mini_batch_size
        :param mini_batch_size:
        :return:
        """
        self.data = np.zeros(mini_batch_size)


class FlappyFlock:

    def __init__(self, size, mini_batch):
        self.size = size
        self.mini_batch = mini_batch
        self.collision_set = CollisionSet(mini_batch)
        # whole group of birds
        self.flock = []
        # mini-batch
        self.birds = []

        self.mini_batch_index = 0

    def update(self, inputs, keys_pressed):
        for bird in self.birds:
            if bird.rect.right >= 0:
                bird.update(inputs, keys_pressed)

    def render(self, window):
        for bird in self.birds:
            if bird.rect.x >= 0:
                bird.render(window)

    def reset(self):
        pass

    def select_batch(self):
        self.birds = self.flock[self.mini_batch_index: min(self.mini_batch_index + self.mini_batch, len(self.flock))]
        self.collision_set.clear(len(self.birds))

    def generate_flappy_birds(self):
        pass

class FlappyBird:

    def __init__(self, brain):
        self.images = FlappyBird.init_images()
        self.current_image = self.images[0]
        self.image_index = 0
        self.brain = brain
        self.rect = pygame.Rect(.2 * SCREEN_SIZE[0], .6 * SCREEN_SIZE[1],
                                self.current_image.get_width(),
                                self.current_image.get_height())
        self.counter = 0
        self.velocity = 0
        self.value = 0
        self.dead = False

    @staticmethod
    def init_images():
        up = pygame.image.load("./assets/sprites/yellowbird-upflap.png").convert_alpha()
        mid = pygame.image.load("./assets/sprites/yellowbird-midflap.png").convert_alpha()
        down = pygame.image.load("./assets/sprites/yellowbird-downflap.png").convert_alpha()
        return [up, mid, down, mid]

    def spawn(self):
        pass

    def reset(self):
        self.current_image = self.images[0]
        self.image_index = 0
        self.rect = pygame.Rect(.2 * SCREEN_SIZE[0], .6 * SCREEN_SIZE[1],
                                self.current_image.get_width(),
                                self.current_image.get_height())
        self.counter = 0
        self.velocity = 0
        self.value = 0

    def score(self, scored):
        if scored:
            self.value += 1

    def update(self, inputs, keys_pressed):
        move = self.brain.step(inputs, keys_pressed)
        self.counter += 1
        move = not self.dead and move
        if move:
            self.velocity = -THRUST
        else:
            self.velocity += GRAVITY
        self.update_pos()
        if self.counter % FLAP_SPEED == 0 and not self.dead:
            self.image_index = 0 if self.image_index + 1 >= len(self.images) else self.image_index + 1
            self.current_image = utils.rot_center(self.images[self.image_index], -self.velocity * 3)

    def update_pos(self):
        if self.rect.y < GAME_HEIGHT:
            self.rect.move_ip(0, self.velocity)
        else:
            self.rect.move_ip(-MAP_MOVEMENT, 0)

    def render(self, window):
        window.blit(self.current_image, self.rect)


class UserFlappyBrain:

    def __init__(self):
        self.name = "User Brain"

    def step(self, inputs, keys_pressed):
        if keys_pressed[K_SPACE] or keys_pressed[K_UP]:
            return True
        return False


class FlappyBrain:

    def __init__(self, driver_id, epsilon):
        # weight initialization range of [-1.5, 1.5)
        self.w1 = 3 * np.random.random((MODEL_INPUTS, 16)) - 1.5
        self.w2 = 3 * np.random.random((16, 8)) - 1.5
        self.w3 = 3 * np.random.random((8, 2)) - 1.5
        self.driver_id = str(driver_id)
        self.epsilon = epsilon

    def forward(self, input_arr: np.array):
        # input shape is (1, num_input)
        pass1 = np.matmul(input_arr, self.w1)  # pass1 has shape of (1, 32)
        pass2 = np.matmul(pass1, self.w2)  # pass2 has shape of (1, 16)
        pass3 = np.matmul(pass2, self.w3)  # pass3 has shape of (1, num_outputs)
        return np.tanh(pass3)  # shape is still (1, num_outputs)

    def step(self, inputs, keys_pressed):
        """
        - Given input from the simulation make a decision
        :param wall_collision: whether the car collided with the wall
        :param reward_collision: whether the car collided with a reward
        :param inputs: sensor input as a numpy array
        :param keys_pressed: a map of pressed keys
        :return direction: int [0 - num_outputs)
        """
        return bool(np.argmax(self.forward(inputs)))

    def cross_over_mutation(self, other, total_batch_size):
        """
        1. cross over self and other
        2. generate {total_batch_size - 2 (self and other)} number of mutations
        3. return the list of drivers as a numpy array, with the first two being the parents
        :param other:
        :return:
        """
        child_driver = self.cross_over(other)
        mutations = child_driver.mutate(total_batch_size - 2)
        return np.array([self, other] + mutations)

    def cross_over(self, other):
        """
        Single point crossover
        :param other:
        :return:
        """
        child_driver = FlappyBrain("({} & {})".format(self.driver_id, other.driver_id), self.epsilon)
        cross_over_point = np.random.randint(self.w1.shape[1])
        child_driver.w1 = np.hstack((self.w1[:, :cross_over_point], other.w1[:, cross_over_point:]))
        cross_over_point = np.random.randint(self.w2.shape[1])
        child_driver.w2 = np.hstack((self.w2[:, :cross_over_point], other.w2[:, cross_over_point:]))
        cross_over_point = np.random.randint(self.w3.shape[1])
        child_driver.w3 = np.hstack((self.w3[:, :cross_over_point], other.w3[:, cross_over_point:]))
        return child_driver

    def mutate(self, num_mutations):
        """
        :param num_mutations: the number of mutations to create
        :return: a list of FlappyBrain mutations
        """
        mutations = []
        for i in range(num_mutations):
            mutation = FlappyBrain(driver_id="{}_{}".format(self.driver_id, i), epsilon=self.epsilon)
            mutation_arr = self.generate_mutation_arr()  # generate mutations and masks
            # read following as add mutations to self.w1 at the positions where the mask is less than epsilon
            mutation.w1 = self.w1 + mutation_arr[0][0] * (mutation_arr[0][1] < self.epsilon)
            mutation.w2 = self.w2 + mutation_arr[1][0] * (mutation_arr[1][1] < self.epsilon)
            mutation.w3 = self.w3 + mutation_arr[2][0] * (mutation_arr[2][1] < self.epsilon)
            mutations.append(mutation)
        return mutations

    def generate_mutation_arr(self):
        """
        :return: list([(mutation array, mask array), ...])
        """
        ret = []
        shapes = [self.w1.shape, self.w2.shape, self.w3.shape]
        for i in shapes:
            # range = number of layers
            ret.append((np.random.random(i) - .5, np.random.random(i)))
        return ret

    @staticmethod
    def generate_drivers(num_drivers, epsilon, load_latest=False):
        """
        create a list of random Drivers
        :param load_latest:
        :param epsilon:
        :param num_drivers:
        :return:
        """
        if load_latest:
            parent = FlappyBrain(driver_id=0, epsilon=epsilon)
            parent.load_model(os.path.join("assets", "models"))
            mutations = parent.mutate(num_drivers - 1)
            return np.array([parent] + mutations)
        return np.array([FlappyBrain(driver_id=identifier, epsilon=epsilon)
                         for identifier in range(num_drivers)])

    def save_model(self, path):
        """
        - Save the brain of the agent to some file (or don't)
        :param path: the path to the model
        :return: None
        """
        path_name = os.path.join(path, "latest_genetic")
        if not os.path.exists(path_name):
            os.mkdir(path_name)
        np.save(os.path.join(path_name, "w1"), self.w1)
        np.save(os.path.join(path_name, "w2"), self.w2)
        np.save(os.path.join(path_name, "w3"), self.w3)

    def load_model(self, path):
        """
        - Load the brain of the agent from some file (or don't)
        :param path: the path to the model
        :return: None
        """
        path_name = os.path.join(path, "latest_genetic")
        self.w1 = np.load(os.path.join(path_name, "w1.npy"))
        self.w2 = np.load(os.path.join(path_name, "w2.npy"))
        self.w3 = np.load(os.path.join(path_name, "w3.npy"))
