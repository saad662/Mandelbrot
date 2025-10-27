import numpy as np
import argparse

from mandel_all import MandelbrotCalculator


class ComputeApp:
    def __init__(self, initial_params=None) -> None:
        # default values
        self._default_iterations = int(100)
        self._default_resolution = int(800)
        self._default_threshold = 2.0
        # self._default_max: float = np.log(default_iterations)
        # self._default_max_interval = (0.0, default_max * 2)
        # self._default_min: float = 0.0
        # self._default_min_interval = (-default_max, default_max)
        self._default_x_min = -2.0
        self._default_y_min = -2.0
        self._default_x_max = 2.0
        self._default_y_max = 2.0

        self._default_use_complex = True
        self._default_in_place = False
        self._default_use_mask = False

        params = initial_params if initial_params else {}

        # current values
        self._current_iterations = params.get("iterations", self._default_iterations)
        self._current_width = params.get("resolution", self._default_resolution)
        self._current_height = params.get("resolution", self._default_resolution)
        self._current_threshold = params.get("threshold", self._default_threshold)
        self._current_x_min = params.get("xmin", self._default_x_min)
        self._current_x_max = params.get("xmax", self._default_x_max)
        self._current_y_min = params.get("ymin", self._default_y_min)
        self._current_y_max = params.get("ymax", self._default_y_max)

        self._current_use_complex = params.get("use_complex", self._default_use_complex)
        self._current_in_place = params.get("in_place", self._default_in_place)
        self._current_use_mask = params.get("use_mask", self._default_use_mask)

        self._active_renderer = MandelbrotCalculator(
            use_complex=self._current_use_complex,
            in_place=self._current_in_place,
            use_mask=self._current_use_mask,
        )

    # getters/setters -----------------------------

    def get_computation_methods(self) -> (bool, bool, bool):
        return (
            self._current_use_complex,
            self._current_in_place,
            self._current_use_mask,
        )

    def set_computation_methods(
        self, use_complex: bool = None, in_place: bool = None, use_mask: bool = None
    ) -> None:
        if use_complex is not None:
            self._current_use_complex = use_complex
        if in_place is not None:
            self._current_in_place = in_place
        if use_mask is not None:
            self._current_use_mask = use_mask

    def get_iterations(self) -> int:
        return self._current_iterations

    def set_iterations(self, iterations: int) -> None:
        if iterations < 2:
            self._current_iterations = 2
        else:
            self._current_iterations = iterations

    def get_resolution(self) -> [int, int]:
        return [self._current_width, self._current_height]

    def set_resolution(self, resolution: [int, int]) -> None:
        if resolution[0] < 1:
            self.current_width = 1  # min: 1x1 pixels
        else:
            self._current_width = resolution[0]
        if resolution[1] < 1:
            self.current_height = 1  # min: 1x1 pixels
        else:
            self._current_height = resolution[1]

    def get_threshold(self) -> float:
        return self._current_threshold

    def set_threshold(self, threshold: float) -> None:
        # TODO validate correctness of value here
        self._current_threshold = threshold

    def get_boundaries(self) -> [float, float, float, float]:
        return [
            self._current_x_min,
            self._current_x_max,
            self._current_y_min,
            self._current_y_max,
        ]

    def set_boundaries(self, bounds: [float, float, float, float]) -> None:
        self._current_x_min = bounds[0]
        self._current_x_max = bounds[1]
        self._current_y_min = bounds[2]
        self._current_y_max = bounds[3]

    def reset_values(self) -> None:
        # set values to default
        self._current_iterations = self._default_iterations
        self._current_width = self._default_resolution
        self._current_height = self._default_resolution
        self._current_threshold = self._default_threshold
        self._current_x_min = self._default_x_min
        self._current_x_max = self._default_x_max
        self._current_y_min = self._default_y_min
        self._current_y_max = self._default_y_max

        self._current_use_complex = self._default_use_complex
        self._current_in_place = self._current_in_place
        self._current_use_mask = self._default_use_mask

    # compute fractal -----------------------------
    def recompute(self, degree) -> np.ndarray:
        # pass values to active renderer (no setters though)
        self._active_renderer.use_complex = self._current_use_complex
        self._active_renderer.in_place = self._current_in_place
        self._active_renderer.use_mask = self._current_use_mask

        # self._active_renderer.set_threshold(self._current_threshold)    #TODO task 3 - how does mandel_all handle threshold value? There is no support for it any more

        current_fractal = self._active_renderer.compute(
            tuple(self.get_resolution()),
            tuple(self.get_boundaries()),
            self.get_iterations(),
            degree,
        )

        return current_fractal


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute App with command line options."
    )

    def str_to_bool(input):
        if isinstance(input, bool):
            return input
        if input.lower() in ("1", "y", "yes", "t", "true"):
            return True
        if input.lower() in ("0", "n", "no", "f", "false"):
            return False
        return argparse.ArgumentTypeError("Boolean value expected")

    parser.add_argument(
        "--use_complex", type=str_to_bool, help=f"Use Complex Mandelbrot. Default: True"
    )
    parser.add_argument(
        "--in_place", type=str_to_bool, help=f"In Place Mandelbrot. Default: False"
    )
    parser.add_argument(
        "--use_mask", type=str_to_bool, help=f"Use Mask Mandelbrot. Default: False"
    )

    parser.add_argument(
        "--resolution",
        type=int,
        help=f"Resolution (width and height) of the fractal image. Default: 800",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        help=f"Minimum X-coordinate for the region of interest. Default: -2.0",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        help=f"Maximum X-coordinate for the region of interest. Default: 2.0",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        help=f"Minimum Y-coordinate for the region of interest. Default: -2.0",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        help=f"Maximum Y-coordinate for the region of interest. Default: 2.0",
    )
    parser.add_argument(
        "--iterations", type=int, help=f"Maximum number of iterations. Default: 100"
    )
    parser.add_argument(
        "--threshold", type=float, help=f"Escape limit (threshold). Default: 2.0"
    )

    parser.add_argument(
        "--output", type=str, help=f"Name of .npy-output file. Default: 'fractal'"
    )

    args = parser.parse_args()

    # Collect provided arguments into a dictionary, skipping None values
    params = {}
    if args.use_complex is not None:
        params["use_complex"] = args.use_complex
    if args.in_place is not None:
        params["in_place"] = args.in_place
    if args.use_mask is not None:
        params["use_mask"] = args.use_mask

    if args.resolution is not None:
        params["resolution"] = args.resolution
    if args.xmin is not None:
        params["xmin"] = args.xmin
    if args.xmax is not None:
        params["xmax"] = args.xmax
    if args.ymin is not None:
        params["ymin"] = args.ymin
    if args.ymax is not None:
        params["ymax"] = args.ymax
    if args.iterations is not None:
        params["iterations"] = args.iterations
    if args.threshold is not None:
        params["threshold"] = args.threshold

    if args.output is not None:
        params["output"] = args.output

    return params


def remove_npy_ending(filename: str) -> str:
    return filename.removesuffix(".npy")


if __name__ == "__main__":
    # parse arguemts
    cmd_params = parse_arguments()
    # init computation with default + parsed arguments
    node_compute = ComputeApp(cmd_params)
    # compute result
    result = node_compute.recompute()
    # save result
    output = cmd_params.get("output", "fractal")
    output = remove_npy_ending(output)  # in case user added file ending themself
    np.save(f"{output}.npy", result)

    print(f"Saved fractal to {output}.npy")
