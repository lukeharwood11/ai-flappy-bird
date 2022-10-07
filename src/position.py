import numpy as np
from constants import GRAVITY


class Position:

    def __init__(self, x=0, y=0, y_velocity=0):
        """
        :param x:
        :param y:
        :param y_velocity:
        """
        self.x = x
        self.y = y

    def move(self, dx=0, dy=0):
        self.x += dx
        self.y += dy
