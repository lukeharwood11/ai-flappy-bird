from __future__ import annotations
import numpy as np
import pygame
from time import time
from utils import calculate_fps
from flappy import FlappyBird, FlappyFlock
from constants import (
    WIDTH_BETWEEN_PIPES,
    HEIGHT_BETWEEN_PIPES,
    STARTING_RANGE,
    MAP_MOVEMENT,
    RANGE_INCREMENT,
    S_HEIGHT,
    S_WIDTH,
    SCORE_Y,
    SCORE_SPACING,
    GAME_HEIGHT,
    PIPE_HEIGHT,
    MAX_RANGE,
)


class Simulator:

    def __init__(self, model, display, window, fps=None):
        self.caption = "AI Flappy Bird!"
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.display = display
        self.model = model
        self.current_timestamp = time()
        self.calc_fps = 0
        self.window = window
        self.game_over = False

    def start(self):
        run = True
        pygame.display.set_caption(self.caption)
        # - - - - - - - - - - - - - - - - - - - - - - - - - -
        while run:
            if self.fps is not None:
                # Use clock for FPS limiting
                self.clock.tick(self.fps)
            else:
                # For maximum speed, don't use clock timing at all
                pygame.event.pump()  # Keep the event system running

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
                self.calc_fps = calculate_fps(max(self.current_timestamp - t, 1))

            keys_pressed = pygame.key.get_pressed()
            run = self.model.update_state(keys_pressed) and run
            self.update_display()
        pygame.quit()

    def update_display(self):
        self.display.render(self.window, self.model, self.game_over)
        pygame.display.update()

    def reset(self):
        pass

    def quit(self):
        pass


class Game:

    def __init__(self, bird):
        """
        Model for the flappy bird game
        """
        # Every 10 levels increase the starting range
        self.pipe_range = STARTING_RANGE
        self.score = 0
        self.height = S_WIDTH
        self.width = S_HEIGHT
        self.bird = bird
        self.pipe_manager = PipeManager(S_WIDTH, S_HEIGHT, self.pipe_range)

    def init(self):
        pass

    def update_state(self, keys_pressed):
        self.pipe_manager.update(self.score)
        self.bird.score(self.handle_score())
        collision = self.handle_collision()
        self.bird.dead = collision
        inputs = self.get_inputs()
        self.bird.update(inputs, keys_pressed)
        if collision:
            self.reset()
            return True
        return True

    def get_inputs(self):
        nearest_pipe = self.pipe_manager.get_current_pipe()
        assert nearest_pipe is not None, "Nearest Pipe is None"
        distance = nearest_pipe.upper_rect.x - self.bird.rect.x
        bird_height = self.bird.rect.centery
        upper = nearest_pipe.upper_rect.bottom
        lower = nearest_pipe.lower_rect.top
        return np.array([distance, bird_height, upper, lower])

    def compare_position(self):
        pass

    def handle_score(self):
        pipe = self.pipe_manager.get_current_pipe()
        if self.bird.rect.x >= pipe.x and not pipe.passed:
            pipe.passed = True
            self.score += 1

    def handle_collision(self):
        return self.pipe_manager.handle_collision(self.bird)

    def reset(self):
        self.bird.reset()
        self.pipe_manager = PipeManager(S_WIDTH, S_HEIGHT, self.pipe_range)
        self.score = 0

    def handle_close_event(self):
        pass


class GeneticGame:

    def __init__(self, flock: FlappyFlock):
        """
        Model for the flappy bird game
        """
        # Every 10 levels increase the starting range
        self.pipe_range: float = STARTING_RANGE
        self.score: int = 0
        self.height: int = S_WIDTH
        self.width: int = S_HEIGHT
        self.flock: FlappyFlock = flock
        self.pipe_manager: PipeManager = PipeManager(S_WIDTH, S_HEIGHT, self.pipe_range)

    def update_state(self, keys_pressed: list[bool]):
        self.flock.update(keys_pressed, self)
        return True

    def reset(self):
        self.pipe_range = STARTING_RANGE
        self.score = 0
        self.height = S_WIDTH
        self.width = S_HEIGHT
        self.pipe_manager = PipeManager(S_WIDTH, S_HEIGHT, self.pipe_range)

    def handle_close_event(self):
        pass


class PipeManager:

    def __init__(
        self, width: int, height: int, starting_range: float, green: bool = True
    ):
        self.pipes: list[PipeSet] = []
        self.top_mask: pygame.mask.Mask = None
        self.bottom_mask: pygame.mask.Mask = None
        self.num_pipes: int = 0
        self.width: int = width
        self.height: int = height
        self.range: float = starting_range
        self.green: bool = green
        self.generate_pipes(0)
        self.init()

    def render(self, window: pygame.Surface):
        for pipe in self.pipes:
            pipe.render(window)

    def init(self):
        s = PipeSet.generate_set(0, 0)
        self.top_mask = pygame.mask.from_surface(s.top_pipe)
        self.bottom_mask = pygame.mask.from_surface(s.bottom_pipe)

    def generate_pipes(self, score: int):
        if score % 10 == 0 and score != 0:
            self.range += RANGE_INCREMENT
            self.range = min(self.range, MAX_RANGE)
        self.pipes.append(PipeSet.generate_set(self.width, self.range))
        self.num_pipes += 1

    def update(self, score: int):
        # check the last pipe to see if we need to generate another set
        if self.width - self.pipes[self.num_pipes - 1].x >= WIDTH_BETWEEN_PIPES:
            self.generate_pipes(score)
        delete = False
        for i, pipe in enumerate(self.pipes):
            delete = pipe.update() or delete
        if delete:
            self.num_pipes -= 1
            self.pipes.pop(0)

    def get_current_pipe(self):
        return self.pipes[0]

    def handle_collision(self, bird: FlappyBird):
        bird_x: int = bird.rect.x
        pipe: PipeSet = self.find_closest_pipe(bird_x)
        bird_mask: pygame.mask.Mask = pygame.mask.from_surface(bird.current_image)

        # check mask
        offset = (bird.rect.x - pipe.upper_rect.x), (bird.rect.y - pipe.upper_rect.y)
        c_upper = self.top_mask.overlap(bird_mask, offset)
        offset = (bird.rect.x - pipe.lower_rect.x), (bird.rect.y - pipe.lower_rect.y)
        c_lower = self.bottom_mask.overlap(bird_mask, offset)

        # Check for collisions
        if (
            c_upper is not None
            or c_lower is not None
            or bird.rect.y >= GAME_HEIGHT
            or bird.rect.y <= 0
        ):
            return True

        # Check if bird is between pipes and colliding with them
        if pipe.in_between(bird_x):
            if (
                bird.rect.bottom > pipe.lower_rect.top
                or bird.rect.top < pipe.upper_rect.bottom
            ):
                return True

        return False

    def find_closest_pipe(self, bird_x: int) -> PipeSet | None:
        for pipe in self.pipes:
            if bird_x < pipe.x or pipe.in_between(bird_x):
                return pipe
        return None


class PipeSet:

    def __init__(self, x: int, y_upper: int, y_lower: int, green: bool = True):
        self.green_pipe: pygame.Surface = pygame.image.load(
            "./src/assets/sprites/pipe-green.png"
        ).convert_alpha()
        self.red_pipe: pygame.Surface = pygame.image.load(
            "./src/assets/sprites/pipe-red.png"
        ).convert_alpha()
        self.bottom_pipe: pygame.Surface = self.green_pipe if green else self.red_pipe
        self.top_pipe: pygame.Surface = pygame.transform.rotate(
            self.bottom_pipe.copy(), 180
        )
        self.x: int = x
        self.upper_rect: pygame.Rect = pygame.Rect(
            x,
            y_upper - self.bottom_pipe.get_height(),
            self.bottom_pipe.get_width(),
            self.bottom_pipe.get_height(),
        )
        self.lower_rect: pygame.Rect = pygame.Rect(
            x, y_lower, self.bottom_pipe.get_width(), self.bottom_pipe.get_height()
        )
        self.passed: bool = False

    def update(self) -> bool:
        delete = False
        self.x -= MAP_MOVEMENT
        self.upper_rect.move_ip(-MAP_MOVEMENT, 0)
        self.lower_rect.move_ip(-MAP_MOVEMENT, 0)
        if self.x + self.lower_rect.width <= 0:
            delete = True
        return delete

    def render(self, window: pygame.Surface):
        window.blit(self.bottom_pipe, self.lower_rect)
        window.blit(self.top_pipe, self.upper_rect)

    def in_between(self, x: int) -> bool:
        x2: int = self.x + self.bottom_pipe.get_width()
        return x2 > x > self.x

    @staticmethod
    def generate_set(width: int, pipe_range: float) -> PipeSet:
        """
        Generate a new pipe set with proper bounds checking
        :return: a top Pipe and a lower pipe
        """
        r: float = np.random.rand() * pipe_range
        r = (-r if np.random.rand() >= 0.4 else r) * (0.4 * GAME_HEIGHT)
        x: int = width

        # Calculate initial positions
        center = 0.5 * GAME_HEIGHT
        top_pipe: float = center + r - (0.5 * HEIGHT_BETWEEN_PIPES)
        bottom_pipe: float = center + r + (0.5 * HEIGHT_BETWEEN_PIPES)

        # Add bounds checking
        pipe_height = PIPE_HEIGHT  # Height of the pipe sprite

        # Ensure top pipe isn't too high (leaving at least 10px from top)
        min_top = 10
        if top_pipe < min_top:
            offset = min_top - top_pipe
            top_pipe += offset
            bottom_pipe += offset

        # Ensure bottom pipe isn't too low (leaving at least 10px from bottom)
        max_bottom = GAME_HEIGHT - 10
        if bottom_pipe > max_bottom:
            offset = bottom_pipe - max_bottom
            top_pipe -= offset
            bottom_pipe -= offset

        return PipeSet(x, top_pipe, bottom_pipe)


class GameBoard:

    def __init__(self, board_width: int, day: bool = True):
        self.background_day: pygame.Surface = pygame.image.load(
            "./src/assets/sprites/background-day.png"
        ).convert()
        self.background_night: pygame.Surface = pygame.image.load(
            "./src/assets/sprites/background-night.png"
        ).convert()
        self.game_over: pygame.Surface = pygame.image.load(
            "./src/assets/sprites/gameover.png"
        ).convert_alpha()
        self.start_message: pygame.Surface = pygame.image.load(
            "./src/assets/sprites/message.png"
        ).convert_alpha()
        self.bg: pygame.Surface = self.background_day if day else self.background_night
        self.ground: Ground = Ground(board_width)
        self.game_height: int = self.bg.get_height() - self.ground.ground.get_width()
        self.score_board: ScoreBoard = ScoreBoard(0, 0)

    def render(self, window: pygame.Surface, model: Game, game_over: bool):
        size: tuple[int, int] = window.get_size()
        window.blit(self.bg, (0, 0))
        ground, x = self.ground.step()
        model.pipe_manager.render(window)
        window.blit(ground, (-x, self.bg.get_height() - ground.get_height()))
        self.score_board.render(window, model.score)
        model.bird.render(window)


class GeneticGameBoard(GameBoard):

    def __init__(self, board_width: int, day: bool = True):
        super().__init__(board_width, day=day)

    def render(self, window: pygame.Surface, model: GeneticGame, game_over: bool):
        window.blit(self.bg, (0, 0))
        ground, x = self.ground.step()
        model.pipe_manager.render(window)
        window.blit(ground, (-x, self.bg.get_height() - ground.get_height()))
        self.score_board.render(window, model.score)
        model.flock.render(window)


class Ground:

    def __init__(self, board_width):
        self.x = 0
        self.board_width = board_width
        self.ground = pygame.image.load("./src/assets/sprites/base.png").convert()

    def step(self) -> tuple[pygame.Surface, int]:
        if self.x + MAP_MOVEMENT >= (self.ground.get_width() - self.board_width) - 10:
            self.x = 0
        self.x += MAP_MOVEMENT
        return self.ground, self.x


class ScoreBoard:

    def __init__(self, x: int, y: int, scale: float = 1):
        """
        :param x:
        :param y:
        :param scale:
        :param height:
        :param width:
        """
        self.scale: float = scale
        self.x: int = x
        self.y: int = y
        self.number_images: list[pygame.Surface] = self.init_number_images()

    def init_number_images(self) -> list[pygame.Surface]:
        images: list[pygame.Surface] = []
        for i in range(10):
            image: pygame.Surface = pygame.image.load(
                "./src/assets/sprites/{}.png".format(i)
            ).convert_alpha()
            size: tuple[int, int] = image.get_size()
            images.append(
                pygame.transform.smoothscale(
                    image, (size[0] * self.scale, size[1] * self.scale)
                )
            )
        return images

    def render(self, window: pygame.Surface, score: int):
        images: list[pygame.Surface] = self.get_images(score)
        t_width: int = 0
        for image in images:
            t_width += image.get_width()
        t_width += SCORE_SPACING * len(images)
        offset: int = (S_WIDTH - t_width) * 0.5
        for image in images:
            window.blit(image, (offset, SCORE_Y))
            offset += image.get_width() + SCORE_SPACING

    def get_images(self, score: int) -> list[pygame.Surface]:
        score: str = str(score)
        ret: list[pygame.Surface] = []
        for char in score:
            ret.append(self.number_images[int(char)])
        return ret
