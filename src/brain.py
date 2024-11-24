from __future__ import annotations
from abc import ABC, abstractmethod
from pygame import K_SPACE, K_UP
import numpy as np


class BaseBrain(ABC):
    """Abstract class representing the decision making process for a FlappyBird"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def step(self, inputs: np.array, keys_pressed: list[bool]) -> bool:
        pass


class BaseGeneticBrain(BaseBrain):

    @abstractmethod
    def cross_over_mutation(self, other: BaseBrain, total_batch_size: int) -> np.array:
        pass

    @staticmethod
    @abstractmethod
    def generate_brains(
        num_brains: int, epsilon: float, load_latest: bool = False
    ) -> np.array:
        pass


class UserFlappyBrain(BaseBrain):

    def __init__(self):
        super().__init__("User Brain")

    def step(self, inputs, keys_pressed):
        if keys_pressed[K_SPACE] or keys_pressed[K_UP]:
            return True
        return False
