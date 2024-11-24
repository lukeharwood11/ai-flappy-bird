from constants import SCREEN_SIZE, FPS
from genetic_brain import FlappyBrain
from brain import UserFlappyBrain
import simulator
import argparse
import pygame
import flappy


def regular():
    pygame.init()
    window = pygame.display.set_mode(SCREEN_SIZE)
    brain = UserFlappyBrain()
    bird = flappy.FlappyBird(brain)
    model = simulator.Game(bird)
    display = simulator.GameBoard(SCREEN_SIZE[0], True)
    sim = simulator.Simulator(model, display, window, fps=60)
    sim.start()


def genetic():
    pygame.init()
    window = pygame.display.set_mode(SCREEN_SIZE)
    # 1000 birds, 1000 mini-batch size, FlappyBrain type
    # create a mini-batch smaller than the total size if you want more birds than your computer can handle
    flock = flappy.FlappyFlock(size=1000, mini_batch=1000, brain_type=FlappyBrain)
    model = simulator.GeneticGame(flock)
    display = simulator.GeneticGameBoard(SCREEN_SIZE[0], True)
    sim = simulator.Simulator(model, display, window, fps=FPS)
    sim.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regular", action="store_true")
    args = parser.parse_args()
    if args.regular:
        regular()
    else:
        genetic()
