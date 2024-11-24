import pygame


def calculate_fps(time_elapsed: float) -> int:
    # convert to seconds (from milliseconds)
    t: float = time_elapsed
    # save to attribute
    return round(1 / t)


def rot_center(image: pygame.Surface, angle: float) -> pygame.Surface:
    rotated_image: pygame.Surface = pygame.transform.rotate(image, angle)
    return rotated_image
