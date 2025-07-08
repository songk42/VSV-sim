import numpy as np
from numba import float64
from numba.experimental import jitclass

spec = [
    ('x', float64),
    ('y', float64),
]

@jitclass(spec)
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


    def move_polar(self, r, theta):
        self.x = self.x + r * np.cos(theta)
        self.y = self.y + r * np.sin(theta)