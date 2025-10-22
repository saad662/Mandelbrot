import numpy as np
from coloring.base import ColoringInterface


class LogColoring(ColoringInterface):
    def apply(self, data, Z, mask, iterations, cycles):
        t = np.abs(Z)
        arg = mask * t + (1 - mask) * 2

        arg = np.clip(arg, 1e-10, None)
        log_arg = np.log(arg)
        log_arg = np.clip(log_arg, 1e-10, None)

        tmp = data + 1 + 1.0 / np.log(2.0) * np.log(np.log(2.0) / log_arg)
        return tmp % (iterations / cycles)
