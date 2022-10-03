import numpy as np
import pygame
from time import time
from utils import calculate_fps
from glob import WIDTH_BETWEEN_PIPES, HEIGHT_BETWEEN_PIPES
from position import Position


class Simulator:

    def __init__(self, model, fps=None):
        self.caption = "AI Flappy Bird!"
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.model = model
        self.current_timestamp = time()
        self.calc_fps = 0

    def start(self):
        run = True
        pygame.display.set_caption(self.caption)
        # - - - - - - - - - - - - - - - - - - - - - - - - - -
        while run:
            if self.fps is not None and run:
                self.clock.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    if self.model is not None:
                        self.model.handle_close_event()
                    break
            t = self.current_timestamp
            self.current_timestamp = time()
            if t is not None:
                self.calc_fps = calculate_fps(self.current_timestamp - t)
            keys_pressed = pygame.key.get_pressed()
            run = self.model.update_state(keys_pressed) or run
            self.model.print_current_state()
            self.update_display()

    def update_display(self):
        pass

    def quit(self):
        pass


class Game:

    def __init__(self):
        """
        Model for the flappy bird game
        """

    def update_state(self, keys_pressed):
        pass

    def headless_start(self):
        pass


class Pipe:

    def __init__(self, upper=False):
        self.position = Position()
        self.upper = upper

    def create(self):
        pass

    def update(self):
        pass

    @staticmethod
    def generate_set():
        """
        :return: a top Pipe and a lower pipe
        """
        pass
