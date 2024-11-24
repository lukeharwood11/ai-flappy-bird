class Position:

    def __init__(self, x: int = 0, y: int = 0, y_velocity: float = 0):
        """
        :param x:
        :param y:
        :param y_velocity:
        """
        self.x: int = x
        self.y: int = y

    def move(self, dx: int = 0, dy: int = 0):
        self.x += dx
        self.y += dy
