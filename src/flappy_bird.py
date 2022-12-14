import pygame

import flappy
from constants import SCREEN_SIZE, FPS
import simulator
import utils


def regular():
    pygame.init()
    window = pygame.display.set_mode(SCREEN_SIZE)
    brain = flappy.UserFlappyBrain()
    bird = flappy.FlappyBird(brain)
    model = simulator.Game(bird)
    display = simulator.GameBoard(SCREEN_SIZE[0], True)
    sim = simulator.Simulator(model, display, window, fps=FPS)
    sim.start()


def genetic():
    pygame.init()
    window = pygame.display.set_mode(SCREEN_SIZE)
    model = simulator.GeneticGame(5000, 1000)
    display = simulator.GeneticGameBoard(SCREEN_SIZE[0], True)
    sim = simulator.Simulator(model, display, window, fps=FPS)
    sim.start()


if __name__ == "__main__":
    genetic()
