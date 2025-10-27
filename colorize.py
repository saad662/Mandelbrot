import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

from coloring.plugin_loader import load_coloring_plugin

class ColorizeApp():
    def __init__(self, initial_params=None) -> None:
        # set default values
        
        # coloring plugins
        self._col_alias = [
            "None",
            "Log",
            "Sqrt",
            "Hist"
        ]
        self._col = [
            "none",
            "coloring.log_coloring.LogColoring",
            "coloring.sqrt_coloring.SqrtColoring",
            "coloring.hist_eq.HistogramEquilization"
        ]
        self._default_col_pos = 1

        # color map
        self._pal = [
            "Accent",
            "Accent_r",
            "Blues",
            "Blues_r",
            "BrBG",
            "BrBG_r",
            "BuGn",
            "BuGn_r",
            "BuPu",
            "BuPu_r",
            "CMRmap",
            "CMRmap_r",
            "Dark2",
            "Dark2_r",
            "GnBu",
            "GnBu_r",
            "Greens",
            "Greens_r",
            "Greys",
            "Greys_r",
            "OrRd",
            "OrRd_r",
            "Oranges",
            "Oranges_r",
            "PRGn",
            "PRGn_r",
            "Paired",
            "Paired_r",
            "Pastel1",
            "Pastel1_r",
            "Pastel2",
            "Pastel2_r",
            "PiYG",
            "PiYG_r",
            "PuBu",
            "PuBuGn",
            "PuBuGn_r",
            "PuBu_r",
            "PuOr",
            "PuOr_r",
            "PuRd",
            "PuRd_r",
            "Purples",
            "Purples_r",
            "RdBu",
            "RdBu_r",
            "RdGy",
            "RdGy_r",
            "RdPu",
            "RdPu_r",
            "RdYlBu",
            "RdYlBu_r",
            "RdYlGn",
            "RdYlGn_r",
            "Reds",
            "Reds_r",
            "summer",
            "summer_r",
            "terrain",
            "terrain_r",
            "turbo",
            "turbo_r",
            "twilight",
            "twilight_r",
            "twilight_shifted",
            "twilight_shifted_r",
            "viridis",
            "viridis_r",
            "winter",
            "winter_r",
            "gist_earth",
            "gist_earth_r",
            "gist_heat",
            "gist_heat_r",
            "gist_ncar",
            "gist_ncar_r",
            "gist_rainbow",
            "gist_rainbow_r",
            "gist_stern",
            "gist_stern_r",
            "gist_yarg",
            "gist_yarg_r",
            "hot",
            "hot_r",
            "hsv",
            "hsv_r",
            "inferno",
            "inferno_r",
            "jet",
            "jet_r",
            "magma",
            "magma_r",
            "nipy_spectral",
            "nipy_spectral_r",
            "ocean",
            "ocean_r",
            "pink",
            "pink_r",
            "plasma",
            "plasma_r",
            "prism",
            "prism_r",
            "rainbow",
            "rainbow_r",
            "seismic",
            "seismic_r",
            "spring",
            "spring_r",
            "Spectral",
            "Spectral_r",
            "Wistia",
            "Wistia_r",
            "YlGn",
            "YlGnBu",
            "YlGnBu_r",
            "YlGn_r",
            "YlOrBr",
            "YlOrBr_r",
            "YlOrRd",
            "YlOrRd_r",
            "afmhot",
            "afmhot_r",
            "autumn",
            "autumn_r",
            "brg",
            "brg_r",
            "bwr",
            "bwr_r",
            "cividis",
            "cividis_r",
            "cool",
            "cool_r",
            "coolwarm",
            "coolwarm_r",
            "copper",
            "copper_r",
            "cubehelix",
            "cubehelix_r",
        ]

        self._default_pal_pos = 49

        self._default_cycles = 1.0

        # init current values

        params = initial_params if initial_params else {}

        try:
            self._current_col_pos = self._col_alias.index(params.get('coloring'))
        except ValueError as e:
            self._current_col_pos = self._default_col_pos

        try:
            self._current_pal_pos = self._pal.index(params.get('palette'))
        except ValueError as e:
            self._current_pal_pos = self._default_pal_pos

        self._current_cycles = params.get('cycles', self._default_cycles)

    #getters/setters -----------------------------
    def get_colorings(self)-> list[str]:
        return self._col_alias

    def get_coloring_name(self) -> str:
        return self._col_alias[self._current_col_pos]
    
    def set_coloring_name(self, name: str) -> None:
        try:
            self._current_col_pos = self._col_alias.index(name)
        except ValueError as e:
            return
    #def set_coloring_index(self) -> None:

    def get_palettes(self)-> list[str]:
        return self._pal
    
    def get_palette_name(self) -> str:
        return self._pal[self._current_pal_pos]
    def get_palette_index(self) -> int:
        return self._current_pal_pos

    def set_palette_name(self, name: str) -> None:
        try:
            self._current_pal_pos = self._pal.index(name)
        except ValueError as e:
            return
    def set_palette_index(self, i: int) -> None:
        if i >= 0 and i < len(_pal):
            self._current_pal_pos = i

    def increment_palette_index(self) -> int:
        self._current_pal_pos = (self._current_pal_pos + 1) % len(self._pal)
        return self._current_pal_pos
    def decrement_palette_index(self) -> int:
        self._current_pal_pos = (self._current_pal_pos - 1 + len(self._pal)) % len(self._pal)
        return self._current_pal_pos

    def get_cycles(self) -> float:
        return self._current_cycles
    def set_cycles(self, cycles: float) -> None:
        #TODO validate correctness of value here
        self._current_cycles = cycles

    def reset_values(self) -> None:
        #set values to default
        self._current_col_pos = self._default_col_pos
        self._current_pal_pos = self._default_pal_pos
        self._current_cycles = self._default_cycles

    #colorize fractal -----------------------------
    def modify_fractal(self, fractal: np.ndarray, iterations: int) -> np.ndarray:
        #modify image
        fractal_modified = self._modify_data(fractal, iterations)
        print("Variance:", np.var(fractal_modified / np.max(fractal_modified)))
        #return result
        return fractal_modified

    def _modify_data(self, data: np.ndarray, iterations: int) -> np.ndarray:
        current_color_plugin = self._col[self._current_col_pos]

        print(f"used color plugin: {current_color_plugin}")

        if current_color_plugin == "none":
            return data

        current_Z = data.astype(np.float64)
        current_Z[current_Z == 0] = 1e-10 #prevent log(0)
        current_mask = data < iterations

        try:
            plugin = load_coloring_plugin(current_color_plugin)
            result = plugin.apply(data, current_Z, current_mask, iterations, self._current_cycles)
        except Exception as e:
            print(f"[Plugin Error] {e}")
            result = data

        return result

    def save_image(self, path: str, fractal: np.ndarray, iterations: float, vmin:float = None) -> None:
        #modified_fractal = self._modify_data(fractal, iterations, cycles) #fractal should be modified explicitly by calling modify_fractal first
        plt.imsave(path, fractal, cmap=self.get_palette_name(),vmax=iterations/self._current_cycles, vmin=vmin)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Mandelbrot Explorer with command line options.")

    parser.add_argument("--input", required=True, 
                        help=f"Name of .npy-input file. (mandatory)")
    parser.add_argument("--output", type=str, 
                        help=f"Name of .png-output file. Default: same as input")

    parser.add_argument("--iterations", type=int, 
                        help=f"Maximum number of iterations. Default: 100")
    parser.add_argument("--cycles", type=float, 
                        help=f"Cycles. Default: 1.0")

    parser.add_argument("--coloring", type=str, 
                        help=f"Coloring Method. Default: Log")
    parser.add_argument("--palette", type=str, 
                        help=f"Color map. Default: RdPu_r")

    args = parser.parse_args()

    # Collect provided arguments into a dictionary, skipping None values
    params = {}
    if args.input is not None:
        params['input'] = args.input
    else:
        print(f".npy-input file required. Pass by setting --input my_file.npy")
        sys.exit()
    if args.output is not None:
        params['output'] = args.output

    if args.iterations is not None:
        params['iterations'] = args.iterations
    if args.cycles is not None:
        params['cycles'] = args.cycles
    if args.coloring is not None:
        params['coloring'] = args.coloring
    if args.palette is not None:
        params['palette'] = args.palette

    return params

def remove_file_endings(filename: str) -> str:
    return filename.removesuffix('.npy').removesuffix('.png')

if __name__ == "__main__":
    #parse arguemts
    cmd_params = parse_arguments()
    #init colorization with default + parsed arguments
    node_colorize = ColorizeApp(cmd_params)
    #load data
    input_name = cmd_params.get('input')
    input_name = remove_file_endings(input_name)
    fractal = np.load(f"{input_name}.npy")
    #get iterations from data if not specified
    maxiter = cmd_params.get('iterations', np.max(fractal))
    #modify input
    fractal = node_colorize.modify_fractal(fractal, maxiter)
    #save image
    output_name = cmd_params.get('output', input_name)
    output_name = remove_file_endings(output_name)
    node_colorize.save_image(
        f"{output_name}.png",
        fractal, 
        maxiter#, 
        #vmin=1
    )

    print(f"Saved fractal to {output_name}.png")