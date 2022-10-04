import numpy as np
import pygame
from time import time
from utils import calculate_fps
from glob import WIDTH_BETWEEN_PIPES, HEIGHT_BETWEEN_PIPES, STARTING_RANGE, SCREEN_SIZE
from position import Position


class Simulator:

    def __init__(self, model, fps=None):
        self.caption = "AI Flappy Bird!"
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.model = model
        self.current_timestamp = time()
        self.calc_fps = 0
        self.window = pygame.display.set_mode(SCREEN_SIZE)

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
                        self.quit()
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
        pygame.display.update()

    def quit(self):
        pass


class Game:

    def __init__(self):
        """
        Model for the flappy bird game
        """
        self.pipe_range = STARTING_RANGE

    def update_state(self, keys_pressed):
        pass

    def headless_start(self):
        pass


class Pipe:

    def __init__(self, green=True, upper=False):
        self.green_pipe = pygame.image.load("./assets/sprites/pipe-green.png").convert_alpha()
        self.red_pipe = pygame.image.load("./assets/sprites/pipe-red.png").convert_alpha()
        self.pipe = self.green_pipe if green else self.red_pipe
        self.position = Position()
        self.upper = upper

    def create(self):
        pass

    def update(self):
        pass

    def get_top_pipe_image(self):
        image = self.pipe.copy()
        return pygame.transform.rotate(image, 180)

    def get_bottom_pipe_image(self):
        return self.pipe.copy()

    @staticmethod
    def generate_set():
        """
        :return: a top Pipe and a lower pipe
        """
        pass


class GameBoard:

    def __init__(self, day=True):
        self.background_day = pygame.image.load("./assets/sprites/background-day.png").convert()
        self.background_night = pygame.image.load("./assets/sprites/background-night.png").convert()
        self.ground = pygame.image.load("./assets/sprites/base.png").convert()
        self.game_over = pygame.image.load("./assets/sprites/gameover.png").convert_alpha()
        self.start_message = pygame.image.load("./assets/sprites/message.png").convert_alpha()
        self.bg = self.background_day if day else self.background_night

    def render(self, window):
        pass


class ScoreBoard:

    def __init__(self, x, y, scale=1, height=0, width=0):
        """
        :param x:
        :param y:
        :param scale:
        :param height:
        :param width:
        """
        self.width = width
        self.height = height
        self.scale = scale
        self.x = x
        self.y = y
        self.number_images = self.init_number_images()

    def init_number_images(self):
        images = []
        for i in range(10):
            image = pygame.image.load("{}.png".format(i)).convert_alpha()
            size = image.get_size()
            images.append(pygame.transform.smoothscale(image, (size[0]*self.scale, size[1]*self.scale)))
        return images

    def render(self, window, score):
        offset = self.x
        images = self.get_images(score)
        for image in images:
            window.blit(image, (offset, self.y))
            offset += image.get_width() + 5

    def get_images(self, score):
        score = str(score)
        ret = []
        for char in score:
            ret.append(self.number_images[int(char)])
        return ret
