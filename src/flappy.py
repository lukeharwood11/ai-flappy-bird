import numpy as np
import pygame
import utils
import typing
from brain import BaseBrain, BaseGeneticBrain
from constants import (
    FLAP_SPEED,
    SCREEN_SIZE,
    THRUST,
    GRAVITY,
    GAME_HEIGHT,
    MAP_MOVEMENT,
)

if typing.TYPE_CHECKING:
    import simulator


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
        return np.all(self.data == np.ones(len(self.data)))

    def clear(self, mini_batch_size):
        """
        mini_batch_size can be smaller than self.mini_batch_size
        :param mini_batch_size:
        :return:
        """
        self.data = np.zeros(mini_batch_size)


class FlappyFlock:

    def __init__(self, size: int, mini_batch: int, brain_type: type[BaseGeneticBrain]):
        self.size: int = size
        self.mini_batch: int = mini_batch
        self.collision_set: CollisionSet = CollisionSet(mini_batch)
        # whole group of birds
        self.flock: list[FlappyBird] = []
        # mini-batch
        self.birds: list[FlappyBird] = []
        self.score_index: int = 0
        self.scores: np.array = np.zeros(size)

        # private vars
        self._brain_type: type[BaseGeneticBrain] = brain_type

        self.mini_batch_index: int = 0
        # initialize flock
        self.generate_flappy_birds()
        # select mini-batch
        self.select_batch()

    def update(self, keys_pressed: list[bool], game: "simulator.Game"):
        game.pipe_manager.update(game.score)
        current_pipe: simulator.Pipe = game.pipe_manager.get_current_pipe()
        if self.collision_set.full_collision():
            self.reset()
            game.reset()

        scored = False
        for i, bird in enumerate(self.birds):
            if bird.rect.right >= 0:
                inputs = self.handle_inputs(current_pipe, bird)
                bird.update(inputs, keys_pressed)
                bird.dead = bird.dead or game.pipe_manager.handle_collision(bird)
                if bird.dead:
                    self.collision_set.set_collision(i)
                scored = scored or self.handle_score(current_pipe, bird)
        game.score = game.score + (1 if scored else 0)

    def handle_inputs(self, nearest_pipe, bird):
        distance: float = nearest_pipe.upper_rect.x - bird.rect.x
        bird_height: float = bird.rect.centery
        upper: float = nearest_pipe.upper_rect.bottom
        lower: float = nearest_pipe.lower_rect.top
        return np.array([distance, bird_height, upper, lower])

    def handle_score(self, current_pipe, bird):
        if (
            bird.rect.center >= current_pipe.upper_rect.center
            and current_pipe != bird.last_pipe
        ):
            bird.last_pipe = current_pipe
            bird.value += 1
            if not current_pipe.passed:
                current_pipe.passed = True
                return True
        return False

    def render(self, window):
        for bird in self.birds:
            if bird.rect.x >= 0:
                bird.render(window)

    def reset(self):
        for bird in self.birds:
            self.scores[self.score_index] = bird.value
            self.score_index += 1
            bird.reset()
        if self.mini_batch_index >= self.size:
            self.reset_flock()
        self.select_batch()

    def reset_flock(self):
        i = np.argsort(self.scores)[-2:]
        parent1 = typing.cast(FlappyBird, self.flock[i[0]])
        parent2 = typing.cast(FlappyBird, self.flock[i[1]])
        # print(self.scores[i[0]], self.scores[i[1]])
        brains = parent1.brain.cross_over_mutation(parent2.brain, self.size)
        for brain, bird in zip(brains, self.flock):
            bird.reset()
            bird.brain = brain
        self.score_index = 0
        self.scores = np.zeros(self.size)
        self.mini_batch_index = 0

    def select_batch(self):
        self.birds = self.flock[
            self.mini_batch_index : min(
                self.mini_batch_index + self.mini_batch, len(self.flock)
            )
        ]
        self.mini_batch_index += self.mini_batch
        self.collision_set.clear(len(self.birds))

    def generate_flappy_birds(self):
        """Must generate genetic brains for the flock"""
        brains: list[BaseGeneticBrain] = self._brain_type.generate_brains(
            self.size, 0.5, False
        )
        for brain in brains:
            self.flock.append(FlappyBird(brain))


class FlappyBird:

    def __init__(self, brain: BaseBrain):
        self.images: list[pygame.Surface] = FlappyBird.init_images()
        self.current_image: pygame.Surface = self.images[0]
        self.image_index: int = 0
        self.brain: BaseBrain = brain
        self.rect: pygame.Rect = pygame.Rect(
            0.2 * SCREEN_SIZE[0],
            0.4 * SCREEN_SIZE[1],
            self.current_image.get_width(),
            self.current_image.get_height(),
        )
        self.counter: int = 0
        self.velocity: float = 0
        self.value: int = 0
        self.last_pipe: simulator.Pipe | None = None
        self.dead: bool = False

    @staticmethod
    def init_images():
        up = pygame.image.load(
            "./src/assets/sprites/yellowbird-upflap.png"
        ).convert_alpha()
        mid = pygame.image.load(
            "./src/assets/sprites/yellowbird-midflap.png"
        ).convert_alpha()
        down = pygame.image.load(
            "./src/assets/sprites/yellowbird-downflap.png"
        ).convert_alpha()
        return [up, mid, down, mid]

    def reset(self):
        self.current_image = self.images[0]
        self.image_index = 0
        self.rect = pygame.Rect(
            0.2 * SCREEN_SIZE[0],
            0.4 * SCREEN_SIZE[1],
            self.current_image.get_width(),
            self.current_image.get_height(),
        )
        self.counter = 0
        self.velocity = 0
        self.value = 0
        self.dead = False

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
            self.image_index = (
                0 if self.image_index + 1 >= len(self.images) else self.image_index + 1
            )
            self.current_image = utils.rot_center(
                self.images[self.image_index], -self.velocity * 3
            )

    def update_pos(self):
        if self.rect.y < GAME_HEIGHT:
            self.rect.move_ip(0, self.velocity)
        else:
            self.rect.move_ip(-MAP_MOVEMENT, 0)

    def render(self, window):
        window.blit(self.current_image, self.rect)
