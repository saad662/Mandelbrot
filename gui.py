import sys
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QCheckBox,
    QRadioButton,
    QComboBox,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
)
from PyQt5.QtCore import Qt, QTimer
import matplotlib as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from compute import ComputeApp
from colorize import ColorizeApp
from videos import VideoApp
from utils import my_cmap
from typing import Any
import argparse


class MandelbrotExplorer(QMainWindow):
    def __init__(self, initial_params=None):
        super().__init__()
        self.current_degree = 2
        self.setWindowTitle("Mandelbrot Explorer (PyQt5)")
        self.setGeometry(100, 100, 1200, 800)  # Adjust window size as needed

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.current_fractal = np.ones((1, 1))  # fractal set by compute-node
        self.modified_fractal = None  # fractal set by colorize-node

        # starting params from console get passed to nodes
        params = initial_params if initial_params else {}
        self.node_compute = ComputeApp(params)
        self.node_colorize = ColorizeApp(params)
        self.node_video = VideoApp(self.node_compute, self.node_colorize)

        # For zoom
        self.start_x: float = None
        self.start_y: float = None
        self.end_x: float = None
        self.end_y: float = None
        self.rect_patch = None

        self.roi_history = []
        self.max_history_depth = 10

        self.init_ui()

        self.add_roi_to_history()

        self.recalculate_image()  # Initial image calculation

    def init_ui(self):
        # Left Control Panel
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_panel.setFixedWidth(250)  # Adjust width

        # Matplotlib Figure and Canvas
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")  # Initially turn off axis
        self.main_layout.addWidget(self.control_panel)
        self.main_layout.addWidget(self.canvas)

        self._create_post_processing_group()
        self._create_renderer_group()
        self._create_functionality_group()
        self._create_parameters_group()
        self._create_colormap_group()
        self._create_action_buttons()
        self._create_video_button()
        self._create_custom_iterations_equation_group()
        self.control_layout.addStretch(1)  # Push everything to the top

        # Video editing panel
        self.video_panel = QWidget()
        self.video_layout = QVBoxLayout(self.video_panel)
        self.video_panel.setFixedWidth(250)  # Adjust width
        self._init_video_panel()
        # do not add video_panel to main_layout yet (that gets toggled via video_button)

        # Mouse events for zoom
        self.canvas.mpl_connect("button_press_event", self.on_selector_press)
        self.canvas.mpl_connect("button_release_event", self.handler_click)

    def _create_post_processing_group(self):
        group_box = QGroupBox("Post-Processing")
        layout = QVBoxLayout()

        self.cmode_dropdown = QComboBox()
        self.cmode_dropdown.addItems(self.node_colorize.get_colorings())
        self.cmode_dropdown.setCurrentText(
            self.node_colorize.get_coloring_name()
        )  # Set default
        layout.addWidget(self.cmode_dropdown)

        group_box.setLayout(layout)
        self.control_layout.addWidget(group_box)

        # Connect signals
        self.cmode_dropdown.currentTextChanged.connect(self.handler_cmode_change)

    def _create_renderer_group(self):
        group_box = QGroupBox("Use Complex")
        layout = QVBoxLayout()
        self.complex_yes_radio = QRadioButton("Yes")
        self.complex_no_radio = QRadioButton("No")
        self.complex_yes_radio.setChecked(
            self.node_compute.get_computation_methods()[0]
        )  # Default
        self.complex_no_radio.setChecked(
            not self.node_compute.get_computation_methods()[0]
        )  # Default
        layout.addWidget(self.complex_yes_radio)
        layout.addWidget(self.complex_no_radio)
        group_box.setLayout(layout)
        self.control_layout.addWidget(group_box)

        # Connect signals
        # self.complex_yes_radio.toggled.connect(lambda: self.handler_complex("Yes"))
        # self.complex_no_radio.toggled.connect(lambda: self.handler_complex("No"))

    def _create_functionality_group(self):
        group_box = QGroupBox("Functionality")
        layout = QVBoxLayout()
        self.inplace_checkbox = QCheckBox("Inplace")
        self.masking_checkbox = QCheckBox("Masking")
        self.inplace_checkbox.setChecked(self.node_compute.get_computation_methods()[1])
        self.masking_checkbox.setChecked(self.node_compute.get_computation_methods()[2])
        layout.addWidget(self.inplace_checkbox)
        layout.addWidget(self.masking_checkbox)
        group_box.setLayout(layout)
        self.control_layout.addWidget(group_box)

        # Connect signals
        # self.inplace_checkbox.toggled.connect(self.handler_functionalities)
        # self.masking_checkbox.toggled.connect(self.handler_functionalities)

    def _create_parameters_group(self):
        group_box = QGroupBox("Parameters")
        grid_layout = QGridLayout()

        # Resolution
        current_res = self.node_compute.get_resolution()
        grid_layout.addWidget(QLabel("Resolution:"), 0, 0)
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("W:"))
        self.res_width_input = QSpinBox()
        self.res_width_input.setRange(100, 2000)
        self.res_width_input.setValue(current_res[0])
        res_layout.addWidget(self.res_width_input)
        res_layout.addWidget(QLabel("H:"))
        self.res_height_input = QSpinBox()
        self.res_height_input.setRange(100, 2000)
        self.res_height_input.setValue(current_res[1])
        res_layout.addWidget(self.res_height_input)
        grid_layout.addLayout(res_layout, 1, 0, 1, 2)  # Span two columns

        # ROI
        current_roi = self.node_compute.get_boundaries()
        grid_layout.addWidget(QLabel("ROI:"), 5, 0)
        roi_x_layout = QHBoxLayout()
        roi_y_layout = QHBoxLayout()
        roi_x_layout.addWidget(QLabel("x_min:"))
        self.roi_xmin_input = QDoubleSpinBox()
        self.roi_xmin_input.setRange(-5, 0)
        self.roi_xmin_input.setValue(current_roi[0])
        roi_x_layout.addWidget(QLabel("x_max:"))
        self.roi_xmax_input = QDoubleSpinBox()
        self.roi_xmax_input.setRange(0, 5)
        self.roi_xmax_input.setValue(current_roi[1])
        roi_y_layout.addWidget(QLabel("y_min:"))
        self.roi_ymin_input = QDoubleSpinBox()
        self.roi_ymin_input.setRange(-5, 0)
        self.roi_ymin_input.setValue(current_roi[2])
        roi_y_layout.addWidget(QLabel("y_max:"))
        self.roi_ymax_input = QDoubleSpinBox()
        self.roi_ymax_input.setRange(0, 5)
        self.roi_ymax_input.setValue(current_roi[3])
        roi_x_layout.addWidget(self.roi_xmin_input)
        roi_x_layout.addWidget(self.roi_xmax_input)
        roi_y_layout.addWidget(self.roi_ymin_input)
        roi_y_layout.addWidget(self.roi_ymax_input)

        grid_layout.addLayout(roi_x_layout, 6, 0, 1, 2)
        grid_layout.addLayout(roi_y_layout, 7, 0, 1, 2)

        # Cycles
        grid_layout.addWidget(QLabel("Cycles:"), 2, 0)
        self.cycles_input = QDoubleSpinBox()
        self.cycles_input.setRange(0.1, 100.0)  # Adjust range as needed
        self.cycles_input.setValue(
            self.node_colorize.get_cycles()
        )  # Default cycles #TODO move this to another field, all other parameters are for compute
        grid_layout.addWidget(self.cycles_input, 2, 1)

        # Threshold
        grid_layout.addWidget(QLabel("Threshold:"), 3, 0)
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(0.1, 100.0)  # Adjust range
        self.threshold_input.setValue(self.node_compute.get_threshold())
        grid_layout.addWidget(self.threshold_input, 3, 1)

        # Iterations
        grid_layout.addWidget(QLabel("Iterations:"), 4, 0)
        self.iterations_input = QSpinBox()
        self.iterations_input.setRange(2, 5000)  # Adjust range
        self.iterations_input.setValue(self.node_compute.get_iterations())
        grid_layout.addWidget(self.iterations_input, 4, 1)

        group_box.setLayout(grid_layout)
        self.control_layout.addWidget(group_box)

    def _create_colormap_group(self):
        group_box = QGroupBox("Colormap")
        layout = QVBoxLayout()

        self.cmap_dropdown = QComboBox()
        self.cmap_dropdown.addItems(self.node_colorize.get_palettes())
        self.cmap_dropdown.setCurrentText(
            self.node_colorize.get_palette_name()
        )  # Set default
        layout.addWidget(self.cmap_dropdown)

        nav_layout = QHBoxLayout()
        self.cmap_prev_btn = QPushButton("Prev")
        self.cmap_next_btn = QPushButton("Next")
        nav_layout.addWidget(self.cmap_prev_btn)
        nav_layout.addWidget(self.cmap_next_btn)
        layout.addLayout(nav_layout)

        group_box.setLayout(layout)
        self.control_layout.addWidget(group_box)

        # Connect signals
        self.cmap_dropdown.currentTextChanged.connect(self.handler_cmap_change)
        self.cmap_prev_btn.clicked.connect(self.handler_cmap_prev)
        self.cmap_next_btn.clicked.connect(self.handler_cmap_next)

    def _create_action_buttons(self):
        action_layout = QHBoxLayout()
        self.recompute_btn = QPushButton("Recompute")
        self.redraw_btn = QPushButton("Redraw")
        self.reset_btn = QPushButton("Reset")
        self.back_btn = QPushButton("Back")
        self.back_btn.setEnabled(False)

        action_layout.addWidget(self.recompute_btn)
        action_layout.addWidget(self.redraw_btn)
        action_layout.addWidget(self.reset_btn)
        action_layout.addWidget(self.back_btn)

        self.control_layout.addLayout(action_layout)
        # self.control_layout.addStretch(1) # Push everything to the top

        # Connect buttons
        self.recompute_btn.clicked.connect(self.handler_draw)
        self.redraw_btn.clicked.connect(self.handler_postprocessing)
        self.reset_btn.clicked.connect(self.handler_reset)
        self.back_btn.clicked.connect(self.handler_back)

    def _create_video_button(self):
        # global video_area_shown
        self.video_area_shown = False

        action_layout = QHBoxLayout()
        self.create_video_btn = QPushButton("Video Creation")

        action_layout.addWidget(self.create_video_btn)

        self.control_layout.addLayout(action_layout)

        # Connect buttons
        self.create_video_btn.clicked.connect(self.handler_toggle_video_creation_area)

    def _create_custom_iterations_equation_group(self):
        group_box = QGroupBox("Custom Iterations Equation")

        layout_btns = QHBoxLayout()
        layout_input = QHBoxLayout()
        main_layout = QVBoxLayout()  # Main vertical layout

        # Pre-made buttons
        self.cmap_sq_btn = QPushButton("Square")
        self.cmap_cubic_btn = QPushButton("Cubic")
        self.cmap_quartic_btn = QPushButton("Quartic")

        layout_btns.addWidget(self.cmap_sq_btn)
        layout_btns.addWidget(self.cmap_cubic_btn)
        layout_btns.addWidget(self.cmap_quartic_btn)

        # Custom degree input from user
        self.degree_input = QLineEdit()
        self.degree_input.setPlaceholderText("Enter degree (e.g. 5)")
        self.degree_input.setFixedWidth(100)

        self.custom_degree_btn = QPushButton("Generate")
        layout_input.addWidget(self.degree_input)
        layout_input.addWidget(self.custom_degree_btn)

        # Combine both rows in main layout
        main_layout.addLayout(layout_btns)
        main_layout.addLayout(layout_input)

        group_box.setLayout(main_layout)
        self.control_layout.addWidget(group_box)

        # Connect signals
        self.cmap_sq_btn.clicked.connect(lambda: self.handle_equation_change_draw(2))
        self.cmap_cubic_btn.clicked.connect(lambda: self.handle_equation_change_draw(3))
        self.cmap_quartic_btn.clicked.connect(
            lambda: self.handle_equation_change_draw(4)
        )
        self.custom_degree_btn.clicked.connect(self.handle_custom_degree_input)

    def _init_video_panel(self):

        # boundary preview
        self.rect_boundary = None
        self.render_bounds_show = False

        # self.video_panel

        # Dropdown for video mode selection
        self.render_mode_pos = 0
        self.render_mode = ["Stepwise", "Interpolated"]
        self.render_mode_dropdown = QComboBox()
        self.render_mode_dropdown.addItems(self.render_mode)
        self.render_mode_dropdown.setCurrentText(
            self.render_mode[self.render_mode_pos]
        )  # Set default
        # Dropdown handler
        self.render_mode_dropdown.currentTextChanged.connect(
            self.handler_render_mode_change
        )

        # general settings
        self.render_general_settings_box = QGroupBox("Render Settings")
        self.render_general_settings_gridlayout = QGridLayout()
        # output dir
        # self.output_dir = "./renders"
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("Output directory")
        # self.output_dir_input.setText(self.output_dir)
        self.output_dir_input.setText("renders")
        self.render_general_settings_gridlayout.addWidget(QLabel("Output:"), 0, 0)
        self.render_general_settings_gridlayout.addWidget(self.output_dir_input, 0, 1)
        # frame count
        # self.frame_num = 1
        self.frame_num_input = QSpinBox()
        self.frame_num_input.setRange(1, 5000)  # Adjust range
        # self.frame_num_input.setValue(self.frame_num)
        self.frame_num_input.setValue(1)
        self.render_general_settings_gridlayout.addWidget(QLabel("Frame Count:"), 1, 0)
        self.render_general_settings_gridlayout.addWidget(self.frame_num_input, 1, 1)

        # render start
        self.render_control_box = QGroupBox("Render Control")
        self.render_control_layout = QVBoxLayout()
        # preview boundaries button
        self.render_bounds_btn = QPushButton("Toggle Bounds") # the name gets adjusted  when used to better reflect current funtionality
        self.render_control_layout.addWidget(self.render_bounds_btn)
        # last frame preview
        self.fig_vid = Figure(figsize=(2.5, 2), dpi=100)
        self.canvas_vid = FigureCanvas(self.fig_vid)
        self.ax_vid = self.fig_vid.add_subplot(111)
        self.ax_vid.axis("off")  # Initially turn off axis
        self.render_control_layout.addWidget(self.canvas_vid)
        # preview + render button
        action_layout = QHBoxLayout()
        self.render_preview_btn = QPushButton("Preview")
        action_layout.addWidget(self.render_preview_btn)
        self.render_start_btn = QPushButton("Render")
        action_layout.addWidget(self.render_start_btn)
        self.render_control_layout.addLayout(action_layout)
        # Connect buttons
        self.render_bounds_btn.clicked.connect(self.handler_toggle_render_bounds)
        self.render_preview_btn.clicked.connect(self.handler_render_preview)
        self.render_start_btn.clicked.connect(self.handler_render_frames)

        # render mode settings (inserted in front of render_control)
        self._init_render_settings()

    # gui video creation area -----------------------------

    def _show_video_panel(self):
        # show render bounds on main fig
        self.handler_set_render_bounds(True)

        # panel
        self.main_layout.addWidget(self.video_panel)

        # dropdown
        self.video_layout.addWidget(self.render_mode_dropdown)
        # general settings
        self.render_general_settings_box.setLayout(
            self.render_general_settings_gridlayout
        )
        self.video_layout.addWidget(self.render_general_settings_box)
        # render controls
        self.render_control_box.setLayout(self.render_control_layout)
        self.video_layout.addWidget(self.render_control_box)

        # content (insert before render start)
        match self.render_mode_pos:
            case 0:
                self._show_render_settings_step()
            case 1:
                self._show_render_settings_interpolation()

    def _hide_video_panel(self):
        # hide render bounds on main fig
        self.handler_set_render_bounds(False)

        # panel
        self.main_layout.removeWidget(self.video_panel)

        # dropdown
        self.video_layout.removeWidget(self.render_mode_dropdown)
        self.render_mode_dropdown.setParent(None)

        # general settings
        self.video_layout.removeWidget(self.render_general_settings_box)
        self.render_general_settings_box.setParent(None)

        # control
        self.video_layout.removeWidget(self.render_control_box)
        self.render_control_box.setParent(None)

        # content
        match self.render_mode_pos:
            case 0:
                self._hide_render_settings_step()
            case 1:
                self._hide_render_settings_interpolation()

    def _init_render_settings(self):
        # step controls (value deltas)
        self.render_step_settings_box = QGroupBox("Value Deltas")
        self.render_step_settings_gridlayout = QGridLayout()

        self.render_step_delta_threshold_input = QDoubleSpinBox()
        self.render_step_delta_threshold_input.setRange(-5000, 5000)  # Adjust range
        self.render_step_delta_threshold_input.setValue(0)
        self.render_step_settings_gridlayout.addWidget(QLabel("Threshold:"), 0, 0)
        self.render_step_settings_gridlayout.addWidget(
            self.render_step_delta_threshold_input, 0, 1
        )

        self.render_step_delta_iterations_input = QSpinBox()
        self.render_step_delta_iterations_input.setRange(-200, 200)  # Adjust range
        self.render_step_delta_iterations_input.setValue(0)
        self.render_step_settings_gridlayout.addWidget(QLabel("Iterations:"), 1, 0)
        self.render_step_settings_gridlayout.addWidget(
            self.render_step_delta_iterations_input, 1, 1
        )

        self.render_step_delta_x_input = QDoubleSpinBox()
        self.render_step_delta_x_input.setRange(-200, 200)  # Adjust range
        self.render_step_delta_x_input.setValue(0)
        self.render_step_delta_y_input = QDoubleSpinBox()
        self.render_step_delta_y_input.setRange(-200, 200)  # Adjust range
        self.render_step_delta_y_input.setValue(0)
        self.render_step_settings_gridlayout.addWidget(QLabel("x offset:"), 2, 0)
        self.render_step_settings_gridlayout.addWidget(
            self.render_step_delta_x_input, 2, 1
        )
        self.render_step_settings_gridlayout.addWidget(QLabel("y offset:"), 3, 0)
        self.render_step_settings_gridlayout.addWidget(
            self.render_step_delta_y_input, 3, 1
        )

        self.render_step_delta_scale_input = QDoubleSpinBox()
        self.render_step_delta_scale_input.setRange(0.000001, 50)  # Adjust range
        self.render_step_delta_scale_input.setValue(1)
        self.render_step_delta_scale_input.setSingleStep(0.1)
        self.render_step_settings_gridlayout.addWidget(QLabel("Scale:"), 4, 0)
        self.render_step_settings_gridlayout.addWidget(
            self.render_step_delta_scale_input, 4, 1
        )
        # change boundary preview whenever deltas get touched
        self.render_step_delta_x_input.valueChanged.connect(self.handler_update_render_bounds_display)
        self.render_step_delta_x_input.valueChanged.connect(self.handler_update_render_bounds_display)
        self.render_step_delta_scale_input.valueChanged.connect(self.handler_update_render_bounds_display)

        self.render_step_settings_gridlayout.setRowStretch(
            self.render_step_settings_gridlayout.rowCount(), 1
        )

        # interpolation controls (final frame definition)
        self.render_inter_settings_box = QGroupBox("Final Frame")
        self.render_inter_settings_gridlayout = QGridLayout()

        self.render_inter_settings_gridlayout.addWidget(QLabel("ROI:"), 0, 0)
        self.render_inter_settings_gridlayout.addWidget(QLabel("x min/max:"), 1, 0)
        self.render_inter_settings_gridlayout.addWidget(QLabel("y min/max:"), 2, 0)

        current_roi = self.node_compute.get_boundaries()

        self.render_inter_x_min_input = QDoubleSpinBox()
        self.render_inter_x_min_input.setRange(-5000, 5000)  # Adjust range
        self.render_inter_x_min_input.setValue(current_roi[0])
        self.render_inter_settings_gridlayout.addWidget(
            self.render_inter_x_min_input, 1, 1
        )
        self.render_inter_x_max_input = QDoubleSpinBox()
        self.render_inter_x_max_input.setRange(-5000, 5000)  # Adjust range
        self.render_inter_x_max_input.setValue(current_roi[1])
        self.render_inter_settings_gridlayout.addWidget(
            self.render_inter_x_max_input, 1, 2
        )
        self.render_inter_y_min_input = QDoubleSpinBox()
        self.render_inter_y_min_input.setRange(-5000, 5000)  # Adjust range
        self.render_inter_y_min_input.setValue(current_roi[2])
        self.render_inter_settings_gridlayout.addWidget(
            self.render_inter_y_min_input, 2, 1
        )
        self.render_inter_y_max_input = QDoubleSpinBox()
        self.render_inter_y_max_input.setRange(-5000, 5000)  # Adjust range
        self.render_inter_y_max_input.setValue(current_roi[3])
        self.render_inter_settings_gridlayout.addWidget(
            self.render_inter_y_max_input, 2, 2
        )
        # change boundary preview whenever roi gets touched
        self.render_inter_x_min_input.valueChanged.connect(self.handler_update_render_bounds_display)
        self.render_inter_x_max_input.valueChanged.connect(self.handler_update_render_bounds_display)
        self.render_inter_y_min_input.valueChanged.connect(self.handler_update_render_bounds_display)
        self.render_inter_y_max_input.valueChanged.connect(self.handler_update_render_bounds_display)

        self.render_inter_settings_gridlayout.addWidget(QLabel("Modes:"), 3, 0)
        self.render_inter_settings_gridlayout.addWidget(QLabel("Scale:"), 4, 0)
        self.render_inter_settings_gridlayout.addWidget(QLabel("Transl:"), 5, 0)

        self.render_inter_mode_scale_input = QSpinBox()
        self.render_inter_mode_scale_input.setRange(
            -1, 1
        )  # -1: not applied, 0: linear, 1: exponential
        self.render_inter_mode_scale_input.setValue(1)
        self.render_inter_settings_gridlayout.addWidget(
            self.render_inter_mode_scale_input, 4, 1
        )
        self.render_inter_mode_trans_input = QSpinBox()
        self.render_inter_mode_trans_input.setRange(
            -1, 1
        )  # -1: not applied, 0: linear, 1: scaleDependant
        self.render_inter_mode_trans_input.setValue(1)
        self.render_inter_settings_gridlayout.addWidget(
            self.render_inter_mode_trans_input, 5, 1
        )

        self.render_inter_settings_gridlayout.setRowStretch(
            self.render_inter_settings_gridlayout.rowCount(), 1
        )

    def _show_render_settings_step(self):
        self.render_step_settings_box.setLayout(self.render_step_settings_gridlayout)
        self.video_layout.insertWidget(
            self.video_layout.count() - 1, self.render_step_settings_box
        )

    def _show_render_settings_interpolation(self):
        self.render_inter_settings_box.setLayout(self.render_inter_settings_gridlayout)
        self.video_layout.insertWidget(
            self.video_layout.count() - 1, self.render_inter_settings_box
        )

    def _hide_render_settings_step(self):
        self.video_layout.removeWidget(self.render_step_settings_box)
        self.render_step_settings_box.setParent(None)

    def _hide_render_settings_interpolation(self):
        self.video_layout.removeWidget(self.render_inter_settings_box)
        self.render_inter_settings_box.setParent(None)

    # input readers -----------------------------
    def get_input_iterations(self) -> int:
        return self.iterations_input.value()

    def get_input_resolution(self) -> [int, int]:
        return [self.res_width_input.value(), self.res_height_input.value()]

    def get_input_threshold(self) -> float:
        return float(self.threshold_input.value())

    def get_input_cycles(self) -> float:
        return float(self.cycles_input.value())

    def get_input_boundaries(self) -> [float, float, float, float]:
        return [
            float(self.roi_xmin_input.value()),
            float(self.roi_xmax_input.value()),
            float(self.roi_ymin_input.value()),
            float(self.roi_ymax_input.value()),
        ]

    # handlers ----------------------------- #TODO getters in this section need to be reworked & moved to #input readers

    def handler_postprocessing(self) -> None:
        self.modify_fractal()
        self.set_image()

    def handler_click(self, event):
        """Handler for mouse clicks"""
        if event.button == 3:  # Right-click
            self.handler_reset()

    def handler_cmode_change(self, text):
        """Handler for colormmode (modify_fractal) dropdown change"""
        self.node_colorize.set_coloring_name(text)
        self.handler_postprocessing()  # remodify array, then redraw

    def handler_cmap_prev(self):
        """Handler for previous colormap button"""
        self.node_colorize.decrement_palette_index()
        self.cmap_dropdown.setCurrentText(self.node_colorize.get_palette_name())

    def handler_cmap_next(self):
        """Handler for next colormap button"""
        self.node_colorize.increment_palette_index()
        self.cmap_dropdown.setCurrentText(self.node_colorize.get_palette_name())

    def handler_cmap_change(self, text):
        """Handler for colormap dropdown change"""
        self.node_colorize.set_palette_name(text)
        self.set_image()  # Only redraw, no recompute needed

    def handler_complex(self, label) -> None:
        """Handler for complex radio buttons"""
        if label == "No":
            print("Set new renderer -> No Complex")
            self.active_renderer = MandelNoComplex()
        else:
            print("Set new renderer -> Complex")
            self.active_renderer = MandelComplex()

        self.active_renderer.set_functionality(
            self.get_current_inplace_state(), self.get_current_masking_state()
        )
        self.active_renderer.set_iterations(self.get_current_iterations())
        self.active_renderer.set_threshold(self.get_current_threshold())
        self.active_renderer.set_resolution(
            self.get_current_width(), self.get_current_height()
        )

        self.recalculate_image()

    def handler_functionalities(self) -> None:
        """Handler for functionalities checkboxes"""
        inplace = self.inplace_checkbox.isChecked()
        masking = self.masking_checkbox.isChecked()
        self.active_renderer.set_functionality(inplace, masking)
        self.recalculate_image()  # Recompute because functionality changes calculation

    def get_current_inplace_state(self) -> bool:
        return self.inplace_checkbox.isChecked()

    def get_current_masking_state(self) -> bool:
        return self.masking_checkbox.isChecked()

    def handler_reset(self) -> None:
        """Handler for reset button"""
        # reset compute
        self.node_compute.reset_values()
        self.iterations_input.setValue(self.node_compute.get_iterations())
        self.threshold_input.setValue(self.node_compute.get_threshold())

        default_res = self.node_compute.get_resolution()
        self.res_width_input.setValue(default_res[0])
        self.res_height_input.setValue(default_res[1])

        default_roi = self.node_compute.get_boundaries()
        self.roi_xmin_input.setValue(default_roi[0])
        self.roi_xmax_input.setValue(default_roi[1])
        self.roi_ymin_input.setValue(default_roi[2])
        self.roi_ymax_input.setValue(default_roi[3])

        self.inplace_checkbox.setChecked(False)  # TODO rework
        self.masking_checkbox.setChecked(False)  # TODO rework
        self.complex_yes_radio.setChecked(
            True
        )  # Set active renderer to complex #TODO rework

        self.roi_history = []
        self.add_roi_to_history()

        # reset colorize
        self.node_colorize.reset_values()
        self.cmode_dropdown.setCurrentText(self.node_colorize.get_coloring_name())
        self.cmap_dropdown.setCurrentText(self.node_colorize.get_palette_name())
        self.cycles_input.setValue(self.node_colorize.get_cycles())

        # calculate & display default fractal
        self.recalculate_image()

    def set_all_computation_params(self) -> None:
        self.node_compute.set_computation_methods(
            use_complex=self.complex_yes_radio.isChecked(),
            in_place=self.inplace_checkbox.isChecked(),
            use_mask=self.masking_checkbox.isChecked(),
        )

        self.node_compute.set_resolution(self.get_input_resolution())
        self.node_compute.set_threshold(self.get_input_threshold())
        self.node_compute.set_iterations(self.get_input_iterations())
        self.node_compute.set_boundaries(self.get_input_boundaries())

    def set_reduced_computation_params(self) -> None:  # This method is for setting up a low res preview
        self.node_compute.set_resolution(
            [100, 100]
        )  # we render at a set 100x100 pixels
        self.node_compute.set_threshold(self.get_input_threshold())
        self.node_compute.set_iterations(
            self.get_input_iterations()
            if self.get_input_iterations() < 50
            else int(self.get_input_iterations() / 2)
        )  # iterations get halved
        self.node_compute.set_boundaries(self.get_input_boundaries())

    def handler_draw(self) -> None:
        """Handler for recompute button and parameter changes"""
        # self.add_roi_to_history()
        self.set_all_computation_params()
        self.recalculate_image(self.current_degree)

    def handle_custom_degree_input(self):
        try:
            degree = int(self.degree_input.text())
            if degree < 2:
                raise ValueError("Degree must be >= 2")
            self.current_degree = degree
            self.recalculate_image(degree)
        except ValueError:
            print("Invalid degree input. Please enter a number ≥ 2.")

    def handle_equation_change_draw(self, degree=2) -> None:
        """Handler for recalculating image for various equation (cubic, quartic)"""
        self.current_degree = degree
        self.recalculate_image(degree)

    def modify_fractal(self) -> None:
        self.node_colorize.set_cycles(
            self.get_input_cycles()
        )  # update cycles before modifying
        self.current_fractal_modified = self.node_colorize.modify_fractal(
            self.current_fractal, self.node_compute.get_iterations()
        )

    def recalculate_image(self, degree=2) -> None:
        """Recalculates and displays the image"""
        self.current_fractal = self.node_compute.recompute(degree)
        self.modify_fractal()
        self.set_image()

    def handler_back(self) -> None:
        """Handler for the 'Back' button."""
        if len(self.roi_history) > 1:
            self.roi_history.pop()
            prev_roi = self.roi_history[-1]

            prev_x_min, prev_x_max, prev_y_min, prev_y_max = prev_roi

            # Update GUI
            self.roi_xmin_input.setValue(prev_x_min)
            self.roi_xmax_input.setValue(prev_x_max)
            self.roi_ymin_input.setValue(prev_y_min)
            self.roi_ymax_input.setValue(prev_y_max)

            self.node_compute.set_boundaries(
                [prev_x_min, prev_x_max, prev_y_min, prev_y_max]
            )

            self.recalculate_image(self.current_degree)
            self.update_back_button_state()

    def update_back_button_state(self) -> None:
        """Enables or disables the Back button."""
        self.back_btn.setEnabled(len(self.roi_history) > 1)

    def add_roi_to_history(self) -> None:
        """Adds the current ROI to history."""
        current_roi = self.node_compute.get_boundaries()
        self.roi_history.append(
            (current_roi[0], current_roi[1], current_roi[2], current_roi[3])
        )
        if len(self.roi_history) > self.max_history_depth:
            self.roi_history.pop(0)
        self.update_back_button_state()

    def set_image(self) -> None:
        # close video editing area
        if self.video_area_shown:
            self._hide_video_panel()
            self.video_area_shown = False

        """Refreshes the mpl canvas with current values"""
        self.ax.clear()
        self.ax.axis("off")
        cf = self.current_fractal_modified
        print(cf.min(), cf.max())
        # ax.imshow(current_fractal_modified, cmap=my_cmap, vmin=min_slider.val, vmax=max_slider.val)
        self.node_colorize.set_cycles(
            self.get_input_cycles()
        )  # update cycles before redrawing
        self.ax.imshow(
            cf,
            cmap=self.node_colorize.get_palette_name(),
            vmax=self.node_compute.get_iterations() / self.node_colorize.get_cycles(),
        )
        # ax.imshow(current_fractal_modified, cmap=my_cmap, vmin=0, vmax=get_current_iterations())
        self.fig.canvas.draw_idle()

    def set_preview(self) -> None:
        """Refreshes the preview canvas with current values"""
        self.ax_vid.clear()
        self.ax_vid.axis("off")
        cf = self.current_fractal_modified
        self.node_colorize.set_cycles(
            self.get_input_cycles()
        )  # update cycles before redrawing
        self.ax_vid.imshow(
            cf,
            cmap=self.node_colorize.get_palette_name(),
            vmax=self.node_compute.get_iterations() / self.node_colorize.get_cycles(),
        )
        self.fig_vid.canvas.draw_idle()

    def on_selector_press(self, event):
        if event.button == 1 and event.inaxes == self.ax:
            self.start_x = event.xdata
            self.start_y = event.ydata
            self.rect_patch = None  # Initialize patch for selection rectangle

            self.motion_cid = self.canvas.mpl_connect(
                "motion_notify_event", self.on_selector_move
            )
            self.release_cid = self.canvas.mpl_connect(
                "button_release_event", self.on_selector_release
            )

    def on_selector_move(self, event):
        if event.inaxes == self.ax and self.start_x is not None:
            x = event.xdata
            y = event.ydata

            if self.rect_patch:
                self.rect_patch.remove()
                self.rect_patch = None

            # Draw a new selection box
            if x is not None and y is not None:
                self.rect_patch = self.ax.add_patch(
                    plt.patches.Rectangle(
                        (min(self.start_x, x), min(self.start_y, y)),
                        abs(x - self.start_x),
                        abs(y - self.start_y),
                        facecolor="none",
                        edgecolor="red",
                        linestyle="--",
                        linewidth=1.5,
                    )
                )
                self.fig.canvas.draw_idle()

    def on_selector_release(self, event):
        if event.button == 1 and self.start_x is not None:  # Left click
            self.end_x = event.xdata
            self.end_y = event.ydata

            # Disconnect motion and release events
            self.canvas.mpl_disconnect(self.motion_cid)
            self.canvas.mpl_disconnect(self.release_cid)

            if self.rect_patch:
                self.rect_patch.remove()
                self.rect_patch = None
                self.fig.canvas.draw_idle()

            if (
                self.start_x is not None
                and self.end_x is not None
                and self.start_y is not None
                and self.end_y is not None
                and (
                    abs(self.start_x - self.end_x) > 0.01
                    or abs(self.start_y - self.end_y) > 0.01
                )
            ):

                self.zoom_to_selection()

    def zoom_to_selection(self):

        # Map coordinates to zoom
        x1, y1 = min(self.start_x, self.end_x), min(self.start_y, self.end_y)
        x2, y2 = max(self.start_x, self.end_x), max(self.start_y, self.end_y)

        current_roi = self.node_compute.get_boundaries()
        current_cx_min = current_roi[0]
        current_cx_max = current_roi[1]
        current_cy_min = current_roi[2]
        current_cy_max = current_roi[3]

        current_res = self.node_compute.get_resolution()
        width = current_res[0]
        height = current_res[1]

        # map to complex plane corner points
        new_roi = [
            current_cx_min + (x1 / width) * (current_cx_max - current_cx_min),
            current_cx_min + (x2 / width) * (current_cx_max - current_cx_min),
            current_cy_min + (y1 / height) * (current_cy_max - current_cy_min),
            current_cy_min + (y2 / height) * (current_cy_max - current_cy_min),
        ]

        self.node_compute.set_boundaries(new_roi)

        self.roi_xmin_input.setValue(new_roi[0])
        self.roi_xmax_input.setValue(new_roi[1])
        self.roi_ymin_input.setValue(new_roi[2])
        self.roi_ymax_input.setValue(new_roi[3])

        self.add_roi_to_history()
        print("New ROI:", new_roi)
        self.recalculate_image(self.current_degree)

    # video creation --------------------------------
    def handler_toggle_video_creation_area(self):
        if self.video_area_shown:
            self._hide_video_panel()
            self.video_area_shown = False
        else:
            self._show_video_panel()
            self.video_area_shown = True

    def handler_render_mode_change(self, text):
        """Handler for render_mode dropdown change"""
        match self.render_mode_pos:
            case 0:
                self._hide_render_settings_step()
            case 1:
                self._hide_render_settings_interpolation()

        self.render_mode_pos = self.render_mode.index(text)

        match self.render_mode_pos:
            case 0:
                self._show_render_settings_step()
            case 1:
                self._show_render_settings_interpolation()

        self.handler_update_render_bounds_display()

    def handler_toggle_render_bounds(self):
        if (self.render_bounds_show):
            self.handler_set_render_bounds(False)
        else:
            self.handler_set_render_bounds(True)

    def handler_set_render_bounds(self, bounds_show: bool):
        if (self.render_bounds_show == bounds_show):
            return

        if (bounds_show):
            self.render_bounds_show = True
        else:
            self.render_bounds_show = False
            # remove boundary preview & redraw
            if self.rect_boundary:
                self.rect_boundary.remove()
                self.rect_boundary = None
                self.fig.canvas.draw_idle()

        self.handler_update_render_bounds_display() # display triggers also when show==false, because it updates button description

    def handler_update_render_bounds_display(self):
        if (self.render_bounds_show == False):
            match self.render_mode_pos:
                case 0:
                    self.render_bounds_btn.setText("Show first delta bounds")
                case 1:
                    self.render_bounds_btn.setText("Show final bounds")
            return

        # clear previous
        if self.rect_boundary:
            self.rect_boundary.remove()
            self.rect_boundary = None

        # Draw a new boundary box
        match self.render_mode_pos:
                case 0:
                    self.render_bounds_btn.setText("Hide first delta bounds")
                    
                    current_x_min = float(self.roi_xmin_input.value())
                    current_x_max = float(self.roi_xmax_input.value())
                    current_y_min = float(self.roi_ymin_input.value())
                    current_y_max = float(self.roi_ymax_input.value())
                    current_roi_width = current_x_max - current_x_min
                    current_roi_height = current_y_max - current_y_min
                    current_res_width = self.res_width_input.value()
                    current_res_height = self.res_height_input.value()
                    bound_scale_delta = float(self.render_step_delta_scale_input.value())

                    rect_x_min = (float(self.render_step_delta_x_input.value()) / current_roi_width + (1-bound_scale_delta)/2) * current_res_width
                    rect_y_min = (float(self.render_step_delta_y_input.value()) / current_roi_height + (1-bound_scale_delta)/2) * current_res_height
                    rect_width = bound_scale_delta * current_res_width
                    rect_height = bound_scale_delta * current_res_height


                    self.rect_boundary = self.ax.add_patch(
                        plt.patches.Rectangle(
                            (rect_x_min, rect_y_min),
                            rect_width,
                            rect_height,
                            facecolor="orange",
                            alpha=0.5,
                            edgecolor="orange",
                            linestyle=":",
                            linewidth=1.5,
                        )
                    )
                case 1:
                    self.render_bounds_btn.setText("Hide final bounds")

                    # get corners of boundary box coordinates
                    bound_x_min = float(self.render_inter_x_min_input.value())
                    bound_x_max = float(self.render_inter_x_max_input.value())
                    bound_y_min = float(self.render_inter_y_min_input.value())
                    bound_y_max = float(self.render_inter_y_max_input.value())

                    # translate coordinates into plot dimensions
                    current_x_min = float(self.roi_xmin_input.value())
                    current_x_max = float(self.roi_xmax_input.value())
                    current_y_min = float(self.roi_ymin_input.value())
                    current_y_max = float(self.roi_ymax_input.value())
                    current_roi_width = current_x_max - current_x_min
                    current_roi_height = current_y_max - current_y_min
                    current_res_width = self.res_width_input.value()
                    current_res_height = self.res_height_input.value()

                    rect_x_min = (bound_x_min - current_x_min) / current_roi_width * current_res_width
                    rect_y_min = (bound_y_min - current_y_min) / current_roi_height * current_res_height
                    rect_width = (bound_x_max - bound_x_min) / current_roi_width * current_res_width
                    rect_height = (bound_y_max - bound_y_min) / current_roi_height * current_res_height

                    self.rect_boundary = self.ax.add_patch(
                        plt.patches.Rectangle(
                            (rect_x_min, rect_y_min),
                            rect_width,
                            rect_height,
                            facecolor="orange",
                            alpha=0.5,
                            edgecolor="orange",
                            linestyle=":",
                            linewidth=1.5,
                        )
                    )
        self.fig.canvas.draw_idle()


    
    def handler_render_preview(self):
        # render & display a small preview of the last frame with current settings
        if (
            self.frame_num_input.value() == 1
        ):  # fallback renderer, if only 1 frame is to be rendered
            self.set_reduced_computation_params()

        else:  # render last frame
            match self.render_mode_pos:
                case 0:  # calculate last
                    self.set_all_computation_params()
                    deltas = {}
                    deltas["iterations"] = (
                        self.render_step_delta_iterations_input.value()
                    )
                    deltas["threshold"] = float(
                        self.render_step_delta_threshold_input.value()
                    )
                    deltas["x"] = float(self.render_step_delta_x_input.value())
                    deltas["y"] = float(self.render_step_delta_y_input.value())
                    deltas["scale"] = float(self.render_step_delta_scale_input.value())
                    self.node_video.render_steps_preview(
                        self.frame_num_input.value(), deltas
                    )
                case 1:  # load values from input
                    self.set_reduced_computation_params()
                    self.node_compute.set_boundaries(
                        [
                            float(self.render_inter_x_min_input.value()),
                            float(self.render_inter_x_max_input.value()),
                            float(self.render_inter_y_min_input.value()),
                            float(self.render_inter_y_max_input.value()),
                        ]
                    )
                case _:
                    return

        self.current_fractal = self.node_compute.recompute(self.current_degree)
        self.modify_fractal()
        self.set_preview()

    def handler_render_frames(self):
        """Handler to initiate render process"""

        # fallback renderer, if only 1 frame is to be rendered (no need to make any setup steps)
        if (
            self.frame_num_input.value() == 1
        ):  # fallback renderer, if only 1 frame is to be rendered
            self.set_all_computation_params()
            # coloration params should always be correct, because they get updated onInputChange => no need to set up

            # rerender & display (without adding to history [we assume only minor changes or none at all before rendering is started])
            self.recalculate_image(self.current_degree)

            # save as image
            self.node_colorize.save_image(
                f"{str( self.output_dir_input.text() )}/0.png",
                self.current_fractal_modified,
                self.node_compute.get_iterations(),  # ,
                # vmin=1
            )

        else:
            match self.render_mode_pos:
                case 0:
                    self.set_all_computation_params()
                    deltas = {}
                    deltas["iterations"] = (
                        self.render_step_delta_iterations_input.value()
                    )
                    deltas["threshold"] = float(
                        self.render_step_delta_threshold_input.value()
                    )
                    deltas["x"] = float(self.render_step_delta_x_input.value())
                    deltas["y"] = float(self.render_step_delta_y_input.value())
                    deltas["scale"] = float(self.render_step_delta_scale_input.value())

                    self.node_video.render_steps(
                        str(self.output_dir_input.text()),
                        self.frame_num_input.value(),
                        deltas,
                        self.current_degree #this should be set in 'self.set_all_computation_params()' instead of passed down into recalculate_image
                    )
                case 1:
                    self.set_all_computation_params()
                    params = {}
                    params["mode_scale"] = self.render_inter_mode_scale_input.value()
                    params["mode_trans"] = self.render_inter_mode_trans_input.value()
                    params["final_bounds"] = [
                        float(self.render_inter_x_min_input.value()),
                        float(self.render_inter_x_max_input.value()),
                        float(self.render_inter_y_min_input.value()),
                        float(self.render_inter_y_max_input.value()),
                    ]

                    self.node_video.render_interpolated(
                        str(self.output_dir_input.text()),
                        self.frame_num_input.value(),
                        params,
                        self.current_degree #this should be set in 'self.set_all_computation_params()' instead of passed down into recalculate_image
                    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Mandelbrot Explorer with command line options."
    )

    # compute args

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

    # held by compute but used by colorize (needs to be moved)
    parser.add_argument("--cycles", type=float, help=f"Cycles. Default: 1.0")

    # colorize args
    parser.add_argument("--coloring", type=str, help=f"Coloring Method. Default: Log")
    parser.add_argument("--palette", type=str, help=f"Color map. Default: RdPu_r")

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

    if args.cycles is not None:
        params["cycles"] = args.cycles
    if args.coloring is not None:
        params["coloring"] = args.coloring
    if args.palette is not None:
        params["palette"] = args.palette

    return params


if __name__ == "__main__":
    initial_params = parse_arguments()
    app = QApplication(sys.argv)
    window = MandelbrotExplorer(initial_params)
    window.show()
    sys.exit(app.exec_())
