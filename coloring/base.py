from abc import ABC, abstractmethod
import numpy as np


class ColoringInterface(ABC):
    @abstractmethod
    def apply(
        self,
        data: np.ndarray,
        Z: np.ndarray,
        mask: np.ndarray,
        iterations: int,
        cycles: int,
    ) -> np.ndarray:

        # modify fractal data
        pass
