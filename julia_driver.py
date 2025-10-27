# julia_driver.py
import os
from pathlib import Path
from julia_all import compute_julia
from colorize import ColorizeApp

def map_pixel_to_complex(px, py, width, height, x_min, x_max, y_min, y_max):
    """
    Convert pixel coordinates (px, py) into a complex number c
    within the Mandelbrot view window.
    """
    real = x_min + (px / (width - 1)) * (x_max - x_min)
    imag = y_min + (py / (height - 1)) * (y_max - y_min)
    return complex(real, imag)

def generate_julia_per_pixel(mandel_view,
                             output_dir='output/julia',
                             sample_step=8,
                             julia_res=(256, 256),
                             max_iter=256,
                             escape_radius=4.0):
    """
    Compute Julia sets for each sampled pixel in the Mandelbrot view.

    mandel_view: dict containing width, height, x_min, x_max, y_min, y_max
    sample_step: how many pixels to skip each step (higher = fewer Julia sets)
    julia_res: output size for each Julia image
    """
    os.makedirs(output_dir, exist_ok=True)

    w, h = mandel_view['width'], mandel_view['height']
    x_min, x_max = mandel_view['x_min'], mandel_view['x_max']
    y_min, y_max = mandel_view['y_min'], mandel_view['y_max']

    colorizer = ColorizeApp()
    total = 0

    print(f"Generating Julia sets for every {sample_step} pixels...")

    for py in range(0, h, sample_step):
        for px in range(0, w, sample_step):
            c = map_pixel_to_complex(px, py, w, h, x_min, x_max, y_min, y_max)

            # compute Julia set for this pixel
            it = compute_julia(julia_res[0], julia_res[1],
                               -1.5, 1.5, -1.5, 1.5,
                               c,
                               max_iter=max_iter,
                               escape_radius=escape_radius)

            # modify and colorize the Julia fractal
            fractal_mod = colorizer.modify_fractal(it, max_iter)

            # build output filename and save
            filename = f"julia_px{px}_py{py}_cx{c.real:.5f}_cy{c.imag:.5f}.png"
            full_path = Path(output_dir) / filename
            colorizer.save_image(str(full_path), fractal_mod, max_iter)

            total += 1
            print(f"Saved {filename}")

    print(f"✅ Done! Generated {total} Julia images in '{output_dir}'")
    return total
