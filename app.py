import numpy as np
import matplotlib.pyplot as plt
import threading
import os
from typing import Any
from customslider import CustomSlider
from matplotlib.widgets import (
    CheckButtons,
    RadioButtons,
    Button,
    TextBox,
    RectangleSelector,
)
from ipywidgets import Dropdown
from utils import debounce, my_cmap
from mandelcomplex import MandelComplex
from mandelnocomplex import MandelNoComplex
from mandel_all import MandelbrotCalculator
from coloring.plugin_loader import load_coloring_plugin
from julia_driver import generate_julia_per_pixel
from videos import create_video

# default values
default_iterations = int(100)
default_resolution = int(800)
default_threshold = 2.0
default_max: float = np.log(default_iterations)
default_max_interval = (0.0, default_max * 2)
default_min: float = 0.0
default_min_interval = (-default_max, default_max)
default_x_min = -2.0
default_y_min = -2.0
default_x_max = 2.0
default_y_max = 2.0

# current values
active_renderer = MandelbrotCalculator(use_complex=True, in_place=False, use_mask=False)
current_fractal = np.ones((1, 1))
current_fractal_modified = None
current_x_min = default_x_min
current_x_max = default_x_max
current_y_min = default_y_min
current_y_max = default_y_max

# set default coords

fig, ax = plt.subplots()
fig.subplots_adjust(left=0.2, bottom=0.25)  # make room for controls


# ------------------ post-processing box
# postprocessing ------------------------------------------
def handler_postprocessing(label) -> None:
    """Handler for postprocessing checkboxes"""
    modify_fractal()
    set_image()


position = fig.add_axes([0.05, 0.85, 0.15, 0.1])
position.text(0.0, 1.05, "Coloring Plugins")
labels = ["Log Coloring", "Sqrt Coloring", "Hist Coloring"]
# visibility = [True, False, False]
post = RadioButtons(position, labels, active=0)
post_id = post.on_clicked(handler_postprocessing)


# --- add a global lock/flag to avoid concurrent runs ---
_julia_lock = threading.Lock()
_julia_running = False

# mouse handler -------------------------------------------
def handler_click(event):
    """Handler for mouse clicks"""
    global current_x_max, current_x_min, current_y_max, current_y_min

    if event.button == 3:
        current_x_max = default_x_max
        current_x_min = default_x_min
        current_y_max = default_y_max
        current_y_min = default_y_min
        active_renderer.set_coordinates(
            current_x_min, current_x_max, current_y_min, current_y_max
        )
        recalculate_image()


fig.canvas.mpl_connect("button_press_event", handler_click)
# ------------------------------


def get_current_coloring_mode() -> str:
    """Returns the currently selected coloring mode"""
    return post.value_selected


pal_pos = 49


def handler_up(x):
    global pal_pos
    pal_pos -= 1
    if pal_pos >= len(pal):
        pal_pos = 0

    if pal_pos < 0:
        pal_pos = len(pal) - 1

    cmap_box.set_val(pal[pal_pos])


def handler_down(x):
    global pal_pos
    pal_pos += 1
    if pal_pos >= len(pal):
        pal_pos = 0

    if pal_pos < 0:
        pal_pos = len(pal) - 1

    cmap_box.set_val(pal[pal_pos])


pal = [
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


position = fig.add_axes([0.20, 0.35, 0.1, 0.1])
pal_up = Button(position, "Prev")
position = fig.add_axes([0.20, 0.45, 0.1, 0.1])
pal_down = Button(position, "Next")
pal_up.on_clicked(handler_up)
pal_down.on_clicked(handler_down)


"""
max_slider = CustomSlider(
  fig,
  [0.13, 0.28, 0.02, 0.50],
  "Max",
  default_max_interval[0],
  default_max_interval[1],
  default_max,
  handler_max_slider
)
"""


# enable complex ------------------------------------------
def handler_complex(label) -> None:
    """Handler for complex radio buttons"""
    global active_renderer

    active_renderer = MandelbrotCalculator(
        use_complex=(label != "No"),
        in_place=get_current_inplace_state(),
        use_mask=get_current_masking_state(),
    )


position = fig.add_axes([0.05, 0.05, 0.15, 0.1])
position.text(0.0, 1.05, "Use complex")
radio = RadioButtons(position, ("Yes", "No"))
radio.on_clicked(handler_complex)


# functionalities -----------------------------------------
def handler_functionalities(label) -> None:
    """Handler for functionalities checkboxes"""
    inplace, masking = check.get_status()
    active_renderer.set_functionality(inplace, masking)


position = fig.add_axes([0.21, 0.05, 0.15, 0.1])
position.text(0.0, 1.05, "Functionality")
labels = ["Inplace", "Masking"]
visibility = [False, False]
check = CheckButtons(position, labels, visibility)
check.on_clicked(handler_functionalities)


def get_current_inplace_state() -> bool:
    """Get the current inplace checkbox state"""
    inplace, masking = check.get_status()
    return inplace


def get_current_masking_state() -> bool:
    """Get the current masking checkbox state"""
    inplace, masking = check.get_status()
    return masking


# resolution ----------------------------------------------
position = fig.add_axes([0.37, 0.105, 0.15, 0.045])
position.text(0.0, 1.1, "Resolution")
res_height = TextBox(
    position, "H", default_resolution, label_pad=-0.15, textalignment="center"
)
position = fig.add_axes([0.37, 0.05, 0.15, 0.045])
res_width = TextBox(
    position, "W", default_resolution, label_pad=-0.15, textalignment="center"
)


def get_current_height() -> int:
    """Get the current image height (user input)"""
    try:
        return int(res_height.text)
    except ValueError:
        print(
            f"Invalid user input (Resolution->Height). Defaulting to {default_resolution}."
        )
        return default_resolution


def get_current_width() -> int:
    """Get the current image width (user input)"""
    try:
        return int(res_width.text)
    except ValueError:
        print(
            f"Invalid user input (Resolution->Width). Defaulting to {default_resolution}."
        )
        return default_resolution


# cycles ------------------------------------- AG
position = fig.add_axes([0.37, 0.15, 0.15, 0.045])
default_cycles = 1
cycles_box = TextBox(
    position, "C", default_cycles, label_pad=-0.15, textalignment="center"
)


def get_cycles():
    global cycles
    return float(cycles_box.text)


# threshold -----------------------------------------------
position = fig.add_axes([0.53, 0.105, 0.15, 0.045])
position.text(0.0, 1.1, "Threshold")
threshold = TextBox(position, "", default_threshold, textalignment="center")


def get_current_threshold() -> float:
    """Get the current mandelbrot threshold (user input)"""
    try:
        return float(threshold.text)
    except ValueError:
        print(f"Invalid user input (Threshold). Defaulting to {default_threshold}.")
        return default_threshold


# reset ---------------------------------------------------
def handler_reset(event) -> None:
    """Handler for reset button"""
    global current_x_max, current_x_min, current_y_max, current_y_min, post_id

    current_x_max = default_x_max
    current_x_min = default_x_min
    current_y_max = default_y_max
    current_y_min = default_y_min

    if get_current_inplace_state():
        check.set_active(0)
    if get_current_masking_state():
        check.set_active(1)

    res_height.set_val(default_resolution)
    res_width.set_val(default_resolution)
    threshold.set_val(default_threshold)
    iterations.set_val(default_iterations)
    radio.set_active(0)
    active_renderer.set_coordinates(
        current_x_min, current_x_max, current_y_min, current_y_max
    )

    post.disconnect(post_id)
    mode = get_current_coloring_mode()
    if mode == "Log Coloring":
        post.set_active(0)
    if mode == "Sqrt Coloring":
        post.set_active(1)
    if mode == "Hist Coloring":
        post.set_active(2)
    post_id = post.on_clicked(handler_postprocessing)

    recalculate_image()


position = fig.add_axes([0.53, 0.05, 0.15, 0.045])
reset = Button(position, "Reset")
reset.on_clicked(handler_reset)

# iterations ----------------------------------------------
position = fig.add_axes([0.69, 0.105, 0.15, 0.045])
position.text(0.0, 1.1, "Iterations")
iterations = TextBox(position, "", default_iterations, textalignment="center")


def get_current_iterations() -> int:
    """Get the current iteration count (user input)"""
    try:
        return int(iterations.text)
    except ValueError:
        print(f"Invalid user input (Iterations). Defaulting to {default_iterations}.")
        return default_iterations


# cmap ----------------
position = fig.add_axes([0.85, 0.105, 0.15, 0.045])
position.text(0.0, 1.1, "CMAP")
cmap_box = TextBox(position, "", pal[pal_pos], textalignment="center")


def get_cmap():
    return cmap_box.text


# recompute button -------------------------------------------
def handler_draw(event) -> None:
    """Handler for recompute button"""
    recalculate_image()


position = fig.add_axes([0.69, 0.05, 0.15, 0.045])
recompute = Button(position, "Recompute")
recompute.on_clicked(handler_draw)

# redraw -----------------------------------------------------
position = fig.add_axes([0.85, 0.05, 0.15, 0.045])
redraw = Button(position, "Redraw")
redraw.on_clicked(handler_postprocessing)

# julia-per-point --------------------------------------------
def handler_julia(event) -> None:
    """Compute Julia sets for each point in the current Mandelbrot view (non-blocking)."""

    def run_julia():
        global _julia_running
        try:
            print("=== [JULIA] start worker ===")
            with _julia_lock:
                if _julia_running:
                    print("=== [JULIA] generation already running, skipping new request ===")
                    return
                _julia_running = True

            mandel_view = {
                'width': get_current_width(),
                'height': get_current_height(),
                'x_min': current_x_min,
                'x_max': current_x_max,
                'y_min': current_y_min,
                'y_max': current_y_max
            }

            output_dir = 'output/julia_from_gui'
            os.makedirs(output_dir, exist_ok=True)

            print("=== [JULIA] generating frames now ...")
            generate_julia_per_pixel(
                mandel_view,
                output_dir=output_dir,
                sample_step=80,
                julia_res=(256, 256),
                max_iter=get_current_iterations()
            )
            print("✅ [JULIA] frame generation complete.")

            # Confirm frames exist
            files = [f for f in os.listdir(output_dir) if f.lower().endswith(".png")]
            print(f"=== [JULIA] found {len(files)} PNGs in {output_dir}")

            if len(files) < 2:
                print("⚠️ [JULIA] Too few frames to create video, skipping video step.")
                return

            print("🎞 [JULIA] Creating Julia video (this will block until done)...")
            create_video(
                input_dir=output_dir,
                output_path='output/julia_animation.mp4',
                fps=10
            )
            print("✅ [JULIA] Video created successfully: output/julia_animation.mp4")

            import time
            import platform, subprocess
            print("=== [JULIA] Sleeping 2 seconds to ensure video flush ===")
            time.sleep(2)  # Give OpenCV time to flush I/O buffers

            # Optional: auto-open the video on Windows
            if platform.system() == "Windows":
                try:
                    subprocess.Popen(['start', 'output/julia_animation.mp4'], shell=True)
                except Exception as e:
                    print("⚠️ Could not open video automatically:", e)

        except Exception as e:
            print("⚠️ [JULIA] Unhandled exception during Julia generation:")
            import traceback
            traceback.print_exc()
        finally:
            _julia_running = False
            print("=== [JULIA] worker finished ===")

    # 🚀 Run directly in a background thread (non-daemon)
    t = threading.Thread(target=run_julia, daemon=False)
    t.start()
    print("Started Julia generation worker thread.")

# add the button
position = fig.add_axes([0.85, 0.15, 0.15, 0.045])
julia_btn = Button(position, "Julia per point")
julia_btn.on_clicked(handler_julia)

# main code -----------------------------------------------
def recalculate_image() -> None:
    """Recalculates and displays the image"""
    global current_fractal, current_Z, current_mask
    shape = (get_current_height(), get_current_width())
    bounds = (current_x_min, current_x_max, current_y_min, current_y_max)
    maxiter = get_current_iterations()
    current_fractal = active_renderer.compute(shape, bounds, maxiter)
    current_Z = current_fractal.astype(np.float64)
    current_Z[current_Z == 0] = 1e-10  # prevent log(0)
    current_mask = current_fractal < maxiter

    modify_fractal()
    set_image()


def modify_data(data: Any) -> Any:
    """Modifies data with log if log state is set"""

    mode = get_current_coloring_mode()
    print("Selected Mode:", mode)
    print("orig", data.min(), data.max())
    iterations = get_current_iterations()
    cycles = get_cycles()

    """
  return np.log(np.log(data)) if use_log else data
  """

    mx = iterations / cycles
    print("iter=", iterations, "cycles=", cycles)
    try:
        if mode == "Log Coloring":
            plugin = load_coloring_plugin("coloring.log_coloring.LogColoring")
            print("Applying === LOG")
        elif mode == "Sqrt Coloring":
            plugin = load_coloring_plugin("coloring.sqrt_coloring.SqrtColoring")
            print("Applying === SQRT")
        elif mode == "Hist Coloring":
            plugin = load_coloring_plugin("coloring.hist_eq.HistogramEquilization")
            print("Applying === HIST")
        else:
            raise ValueError("Unknown coloring mode selected.")

        result = plugin.apply(data, current_Z, current_mask, iterations, cycles)
    except Exception as e:
        print(f"[Plugin Error] {e}")
        result = data

    # print("TMP=", tmp.min(), tmp.max())
    return result


def modify_fractal() -> None:
    """Modifies the fractal array if log state is set"""
    global current_fractal_modified
    current_fractal_modified = modify_data(current_fractal)
    print(
        "Variance:", np.var(current_fractal_modified / np.max(current_fractal_modified))
    )


def set_image_maximum():
    """Sets the image maximum value as slider values"""
    current_max = np.max(current_fractal_modified)
    # max_slider.set_val(current_max, True)
    # min_slider.set_val(0.0, True)


def set_image() -> None:
    """Refreshes the mpl canvas with current values"""
    ax.clear()
    ax.axis("off")
    cf = current_fractal_modified
    # AG
    print(cf.min(), cf.max())
    # ax.imshow(current_fractal_modified, cmap=my_cmap, vmin=min_slider.val, vmax=max_slider.val)
    ax.imshow(
        current_fractal_modified,
        cmap=get_cmap(),
        vmax=get_current_iterations() / get_cycles(),
    )
    # ax.imshow(current_fractal_modified, cmap=my_cmap, vmin=0, vmax=get_current_iterations())
    fig.canvas.draw_idle()


# interactive selector ------------------------------------
def handler_selector(eclick, erelease) -> None:
    """Handler for rectangle selector"""
    global current_x_max, current_x_min, current_y_max, current_y_min

    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    width = get_current_width()
    height = get_current_height()

    # map to 0..1
    x1_a: float = x1 / width
    y1_a: float = y1 / height
    x2_a: float = x2 / width
    y2_a: float = y2 / height

    # map to complex plane corner points
    x_min = x1_a * (current_x_max - current_x_min) + current_x_min
    y_min = y1_a * (current_y_max - current_y_min) + current_y_min
    x_max = x2_a * (current_x_max - current_x_min) + current_x_min
    y_max = y2_a * (current_y_max - current_y_min) + current_y_min

    current_x_min = x_min
    current_x_max = x_max
    current_y_min = y_min
    current_y_max = y_max

    # set coordinates
    recalculate_image()


selector = RectangleSelector(
    ax,
    handler_selector,
    useblit=False,
    button=[1],
    minspanx=5,
    minspany=5,
    spancoords="pixels",
    interactive=False,
)


recalculate_image()
plt.show()
