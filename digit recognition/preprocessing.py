import random

import numpy as np

PIXEL_THRESHOLD = 127


def binarize(pixels):
    """Чёрно-белое изображение: каждый пиксель -> 0 или 1."""
    return (pixels > PIXEL_THRESHOLD).astype(np.uint8)


def split_by_class(labels, target):
    """Индексы изображений изучаемого класса и всех остальных."""
    return np.flatnonzero(labels == target), np.flatnonzero(labels != target)


def shuffled(positive, negative, rng=random):
    """Все объекты обеих групп в случайном порядке."""
    order = list(positive) + list(negative)
    rng.shuffle(order)
    return order
