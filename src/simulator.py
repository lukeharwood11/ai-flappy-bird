import numpy as np
import pygame
from time import time
from utils import calculate_fps
from flappy import FlappyBird
from glob import WIDTH_BETWEEN_PIPES, HEIGHT_BETWEEN_PIPES, STARTING_RANGE, SCREEN_SIZE, MAP_MOVEMENT, RANGE_INCREMENT
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

    def __init__(self, width, height, bird):
        """
        Model for the flappy bird game
        """
        # Every 10 levels increase the starting range
        self.pipe_range = STARTING_RANGE
        self.score = 0
        self.pipes = []
        self.height = height
        self.width = width
        self.bird = bird

    def init(self):
        self.pipes.append(PipeSet.generate_set(self.width, self.height, self.pipe_range))

    def update_state(self, keys_pressed):
        for pipe in self.pipes:
            pipe.update()

    def compare_position(self):
        pass

    def score(self):
        self.score += 1

    def reset(self):
        self.bird.reset()


class PipeManager:

    def __init__(self, width, height, starting_range, green=True):
        self.pipes = []
        self.top_mask = None
        self.bottom_mask = None
        self.num_pipes = 0
        self.width = width
        self.height = height
        self.range = starting_range
        self.green = green
        self.generate_pipes(0)

    def init(self):
        s = PipeSet.generate_set(0, 0, 0)
        self.top_mask = pygame.mask.from_surface(s.top_pipe)
        self.bottom_mask = pygame.mask.from_surface(s.bottom_pipe)

    def generate_pipes(self, score):
        if score % 10 and score != 0:
            self.range += RANGE_INCREMENT
        self.pipes.append(PipeSet.generate_set(self.width, self.height, self.range))
        self.num_pipes += 1

    def update(self, score):
        # check the last pipe to see if we need to generate another set
        if self.width - self.pipes[self.num_pipes - 1].x >= WIDTH_BETWEEN_PIPES:
            self.generate_pipes(score)
        for pipe in self.pipes:
            pipe.update()

    def check_for_collision(self, bird):
        bird_x = bird.current_image.get_width()
        pipe = self.find_closest_pipe(bird_x)
        # check mask
        # else check if height is above the cutoff or below the cutoff
        pass


    def find_closest_pipe(self, bird_x):
        for pipe in self.pipes:
            if bird_x < pipe.x or pipe.in_between(bird_x):
                return pipe


class PipeSet:

    def __init__(self, x, y_upper, y_lower, green=True):
        self.green_pipe = pygame.image.load("./assets/sprites/pipe-green.png").convert_alpha()
        self.red_pipe = pygame.image.load("./assets/sprites/pipe-red.png").convert_alpha()
        self.bottom_pipe = self.green_pipe if green else self.red_pipe
        self.top_pipe = pygame.transform.rotate(self.bottom_pipe.copy(), 180)
        self.x = x
        self.upper_rect = pygame.Rect(
            x,
            y_upper - self.bottom_pipe.get_height(),
            self.bottom_pipe.get_width(),
            self.bottom_pipe.get_height()
        )
        self.lower_rect = pygame.Rect(
            x,
            y_lower,
            self.bottom_pipe.get_width(),
            self.bottom_pipe.get_height()
        )

    def create(self):
        pass

    def update(self):
        pass

    def render(self, window):
        pass

    def in_between(self, x):
        x2 = self.x + self.bottom_pipe.get_width()
        return x2 > x > self.x

    @staticmethod
    def generate_set(width, height, pipe_range):
        """
        :return: a top Pipe and a lower pipe
        """
        r = np.random.rand() * pipe_range
        r = (-r if np.random.rand() >= .5 else r) * (.4 * height)
        x = width
        top_pipe = (.5 * height) + r - (.5 * HEIGHT_BETWEEN_PIPES)
        bottom_pipe = (.5 * height) + r + (.5 * HEIGHT_BETWEEN_PIPES)
        return PipeSet(x, top_pipe, bottom_pipe, upper=True)


class GameBoard:

    def __init__(self, board_width, day=True):
        self.background_day = pygame.image.load("./assets/sprites/background-day.png").convert()
        self.background_night = pygame.image.load("./assets/sprites/background-night.png").convert()
        self.game_over = pygame.image.load("./assets/sprites/gameover.png").convert_alpha()
        self.start_message = pygame.image.load("./assets/sprites/message.png").convert_alpha()
        self.bg = self.background_day if day else self.background_night
        self.ground = Ground(board_width)
        self.game_height = self.bg.get_height() - self.ground.ground.get_width()

    def render(self, window, game_over):
        size = window.get_size()
        window.blit(self.bg, (0, 0))
        ground, x = self.ground.step()
        window.blit(ground, (-x, self.bg.get_height() - ground.get_height()))


class Ground:

    def __init__(self, board_width):
        self.x = 0
        self.board_width = board_width
        self.ground = pygame.image.load("./assets/sprites/base.png").convert()

    def step(self):
        if self.x + MAP_MOVEMENT >= (self.ground.get_width() - self.board_width):
            self.x = 0
        self.x += MAP_MOVEMENT
        return self.ground, -self.x


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
            images.append(pygame.transform.smoothscale(image, (size[0] * self.scale, size[1] * self.scale)))
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
