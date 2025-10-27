from mandel_all import MandelbrotCalculator
import matplotlib.pyplot as plt

# Parameters for the Mandelbrot set
shape = (400, 600)  # height, width
bounds = (-2.0, 1.0, -1.0, 1.0)  # x_min, x_max, y_min, y_max
maxiter = 100

# Test all 4 combinations (complex yes/no, in-place yes/no)
configs = [
    {"use_complex": True, "in_place": False},
    {"use_complex": True, "in_place": True},
    {"use_complex": False, "in_place": False},
    {"use_complex": False, "in_place": True}
]

for config in configs:
    print(f"Testing config: {config}")
    calc = MandelbrotCalculator(**config)
    result = calc.compute(shape, bounds, maxiter)

    # Check result type and shape
    assert result.shape == shape, f"Shape mismatch: expected {shape}, got {result.shape}"
    assert result.dtype in ["int32", "int64", "float64", "float32"], f"Unexpected dtype: {result.dtype}"

    # Show result briefly
    plt.imshow(result, cmap="turbo")
    plt.title(f"Config: {config}")
    plt.colorbar()
    plt.show()
