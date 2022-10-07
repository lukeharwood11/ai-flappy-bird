import pygame


def calculate_fps(time_elapsed):
    # convert to seconds (from milliseconds)
    t = time_elapsed
    # save to attribute
    return round(1 / t)


def rot_center(image, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    return rotated_image
