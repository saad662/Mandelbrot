# julia_all.py
import numpy as np

def compute_julia(width, height, x_min, x_max, y_min, y_max, c, max_iter=256, escape_radius=4.0):
    """
    Compute the Julia set for a fixed complex constant c.
    Returns a 2D numpy array of iteration counts (shape: (height, width)).
    """
    xs = np.linspace(x_min, x_max, width, dtype=np.float64)
    ys = np.linspace(y_min, y_max, height, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    z = X + 1j * Y

    it = np.zeros(z.shape, dtype=np.int32)
    mask = np.ones(z.shape, dtype=bool)

    for k in range(max_iter):
        z[mask] = z[mask] * z[mask] + c
        escaped = np.abs(z) > escape_radius
        newly_escaped = escaped & mask
        it[newly_escaped] = k
        mask &= ~newly_escaped
        if not mask.any():
            break

    it[mask] = max_iter
    return it
