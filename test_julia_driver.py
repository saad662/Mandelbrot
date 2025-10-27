from julia_driver import generate_julia_per_pixel

mandel_view = {
    'width': 200,
    'height': 150,
    'x_min': -2.0,
    'x_max': 1.0,
    'y_min': -1.2,
    'y_max': 1.2
}

generate_julia_per_pixel(
    mandel_view,
    output_dir='output/julia_test',
    sample_step=40,
    julia_res=(256, 256),
    max_iter=256
)
