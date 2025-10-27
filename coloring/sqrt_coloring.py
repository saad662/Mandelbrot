from coloring.base import ColoringInterface
import numpy as np


class SqrtColoring(ColoringInterface):
    def apply(self, data, Z, mask, iterations, cycles):
        transformed = np.sqrt(iterations + 2 - data)
        l, h = transformed.min(), transformed.max()
        normalized = (transformed - l) / (h - l)
        return (normalized * iterations) % (iterations / cycles)
