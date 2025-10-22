import tensorflow as tf


class MandelbrotCalculator:
    def __init__(self, use_complex=True, in_place=False, use_mask=False):
        self.use_complex = use_complex
        self.in_place = in_place
        self.use_mask = use_mask

    def compute(self, *args, **kwargs):
        if self.use_complex:
            if self.in_place:
                return self._compute_complex_in_place(*args, **kwargs)
            else:
                return self._compute_complex(*args, **kwargs)
        else:
            if self.in_place:
                return self._compute_nocomplex_in_place(*args, **kwargs)
            else:
                return self._compute_nocomplex(*args, **kwargs)

    def _compute_complex(self, shape, bounds, maxiter, degree=2):
        import numpy as np

        height, width = shape
        x_min, x_max, y_min, y_max = bounds
        y, x = np.ogrid[y_min : y_max : height * 1j, x_min : x_max : width * 1j]
        c = x + 1j * y
        z = np.zeros_like(c)
        div_time = maxiter + np.zeros(z.shape, dtype=int)

        for i in range(maxiter):
            z = z**degree + c
            mask = np.abs(z) > 2
            div_now = mask & (div_time == maxiter)
            div_time[div_now] = i
            z[mask] = 2

        return div_time

    def _compute_complex_in_place(self, shape, bounds, maxiter, degree=2):
        import numpy as np

        height, width = shape
        x_min, x_max, y_min, y_max = bounds
        y, x = np.ogrid[y_min : y_max : height * 1j, x_min : x_max : width * 1j]
        c = x + 1j * y
        z = np.zeros_like(c)
        div_time = np.full(z.shape, maxiter, dtype=int)

        if self.use_mask:
            # Only keep updating pixels that haven't diverged yet
            mask = np.full(z.shape, True, dtype=bool)
            for i in range(maxiter):
                z[mask] = z[mask] ** degree + c[mask]
                mask_now = np.abs(z) > 2
                newly_diverged = mask_now & (div_time == maxiter)
                div_time[newly_diverged] = i
                mask &= ~mask_now
                if not mask.any():
                    break
        else:
            for i in range(maxiter):
                z = z * z + c
                mask_now = np.abs(z) > 2
                newly_diverged = mask_now & (div_time == maxiter)
                div_time[newly_diverged] = i
                z[mask_now] = 2  # To keep z stable

        return div_time

    def _compute_nocomplex(self, shape, bounds, maxiter, degree=2):
        import numpy as np

        height, width = shape
        x_min, x_max, y_min, y_max = bounds
        x = np.linspace(x_min, x_max, width)
        y = np.linspace(y_min, y_max, height)
        X, Y = np.meshgrid(x, y)

        Zx = np.zeros_like(X)
        Zy = np.zeros_like(Y)
        Cx = X
        Cy = Y
        div_time = maxiter + np.zeros(X.shape, dtype=int)

        for i in range(maxiter):
            Zx2 = Zx * Zx
            Zy2 = Zy * Zy
            Zxy = Zx * Zy

            # Mandelbrot update rule without complex numbers
            if degree == 2:
                Zx_new = Zx2 - Zy2 + Cx
                Zy_new = 2 * Zxy + Cy
            elif degree == 3:
                # (Zx + iZy)^3 = Zx^3 - 3ZxZy^2 + i(3Zx^2Zy - Zy^3)
                Zx_new = Zx**3 - 3 * Zx * Zy**2 + Cx
                Zy_new = 3 * Zx**2 * Zy - Zy**3 + Cy
            elif degree == 4:
                # (Zx + iZy)^4 = Zx^4 - 6Zx^2Zy^2 + Zy^4 + i(4Zx^3Zy - 4ZxZy^3)
                Zx_new = Zx**4 - 6 * Zx**2 * Zy**2 + Zy**4 + Cx
                Zy_new = 4 * Zx**3 * Zy - 4 * Zx * Zy**3 + Cy
            else:
                raise ValueError("Unsupported degree")
            Zx = Zx_new
            Zy = Zy_new

            mag = Zx2 + Zy2
            mask = mag > 4
            div_now = mask & (div_time == maxiter)
            div_time[div_now] = i

            # Avoid growing values
            Zx[mask] = 2
            Zy[mask] = 2

        return div_time

    def _compute_nocomplex_in_place(self, shape, bounds, maxiter, degree=2):
        import numpy as np

        height, width = shape
        x_min, x_max, y_min, y_max = bounds
        x_vals = np.linspace(x_min, x_max, width)
        y_vals = np.linspace(y_min, y_max, height)
        X, Y = np.meshgrid(x_vals, y_vals)
        Zx = X.copy()
        Zy = Y.copy()
        Cx = X
        Cy = Y
        div_time = np.full(Zx.shape, maxiter, dtype=int)

        if self.use_mask:
            mask = np.full(Zx.shape, True, dtype=bool)
            for i in range(maxiter):
                Zx2 = Zx * Zx
                Zy2 = Zy * Zy
                abs_squared = Zx2 + Zy2

                mask_now = abs_squared > 4
                newly_diverged = mask_now & (div_time == maxiter)
                div_time[newly_diverged] = i
                mask &= ~mask_now

                if not mask.any():
                    break

                Zx_m = Zx[mask]
                Zy_m = Zy[mask]
                Cx_m = Cx[mask]
                Cy_m = Cy[mask]

                if degree == 2:
                    Zx_new = Zx_m**2 - Zy_m**2 + Cx_m
                    Zy_new = 2 * Zx_m * Zy_m + Cy_m
                elif degree == 3:
                    Zx_new = Zx_m**3 - 3 * Zx_m * Zy_m**2 + Cx_m
                    Zy_new = 3 * Zx_m**2 * Zy_m - Zy_m**3 + Cy_m
                elif degree == 4:
                    Zx_new = Zx_m**4 - 6 * Zx_m**2 * Zy_m**2 + Zy_m**4 + Cx_m
                    Zy_new = 4 * Zx_m**3 * Zy_m - 4 * Zx_m * Zy_m**3 + Cy_m
                else:
                    raise ValueError("Unsupported degree")

                Zx[mask] = Zx_new
                Zy[mask] = Zy_new

        else:
            for i in range(maxiter):
                if degree == 2:
                    Zx_new = Zx**2 - Zy**2 + Cx
                    Zy_new = 2 * Zx * Zy + Cy
                elif degree == 3:
                    Zx_new = Zx**3 - 3 * Zx * Zy**2 + Cx
                    Zy_new = 3 * Zx**2 * Zy - Zy**3 + Cy
                elif degree == 4:
                    Zx_new = Zx**4 - 6 * Zx**2 * Zy**2 + Zy**4 + Cx
                    Zy_new = 4 * Zx**3 * Zy - 4 * Zx * Zy**3 + Cy
                else:
                    raise ValueError("Unsupported degree")
                abs_squared = Zx2 + Zy2
                mask_now = abs_squared > 4
                newly_diverged = mask_now & (div_time == maxiter)
                div_time[newly_diverged] = i

                Zx_new = Zx * Zx - Zy * Zy + Cx
                Zy_new = 2 * Zx * Zy + Cy

                Zx[~mask_now] = Zx_new[~mask_now]
                Zy[~mask_now] = Zy_new[~mask_now]

        return div_time
