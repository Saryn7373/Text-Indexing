"""Нейрон: хранит вес одного пикселя."""
import random


class Neuron:
    def __init__(self, weight=None):
        self.weight = random.uniform(-0.1, 0.1) if weight is None else weight

    def calculate(self, value):
        """Вход пикселя (0 или 1) умножается на вес."""
        return self.weight * value

    def update(self, delta):
        self.weight += delta
