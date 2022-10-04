import numpy as np
import pygame


class FlappyBird:

    def __init__(self, brain):
        self.images = FlappyBird.init_images()
        self.current_image = self.images[0]
        self.image_count = 0
        self.brain = brain

    @staticmethod
    def init_images():
        up = pygame.image.load("./assets/sprites/yellowbird-upflap.png").convert_alpha()
        mid = pygame.image.load("./assets/sprites/yellowbird-midflap.png").convert_alpha()
        down = pygame.image.load("./assets/sprites/yellowbird-downflap.png").convert_alpha()
        return [up, mid, down, mid]

    def spawn(self):
        pass

    def update(self):
        pass


class FlappyBrain:

    def __init__(self):
        """
        Bird brain which implements the genetic algorithm
        """


    def update(self):
        pass