from compute import compute_grid
from colorize import colorize_julia

# 1. Compute Julia data
arr = compute_grid('julia', -1.5, 1.5, -1.0, 1.0, 800, 600, 256, c0=complex(-0.4, 0.6))

# 2. Colorize and save
colorize_julia(arr, cmap_name="plasma", output_path="julia_test.png", cycles=2)
