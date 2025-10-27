import numpy as np
from coloring.base import ColoringInterface


class HistogramEquilization(ColoringInterface):

    def apply(self, data, Z, mask, iterations, cycles):
        hist = np.bincount(data.flatten().astype("int32"), minlength=iterations)
        hist = hist / hist.sum()
        chist = np.cumsum(hist)
        res = chist[data.ravel().astype("int32")].reshape(data.shape) * iterations

        return res % (iterations / cycles)
