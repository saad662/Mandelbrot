import numpy as np
import cv2
import os
from compute import ComputeApp
from colorize import ColorizeApp
from natsort import natsorted

class VideoApp():
    def __init__(self, node_compute: ComputeApp, node_colorize: ColorizeApp) -> None:
        self.node_compute = node_compute
        self.node_colorize = node_colorize

    #the render methods assume that the inputs for the initial frame are already applied
    def render_steps(self, output_dir: str, frame_count: int, deltas, degree=2) -> None:
        #Init unchanging variables

        #self.active_renderer.set_resolution(self.get_current_height(), self.get_current_width())
        r_output_dir = output_dir
        r_frame_count = frame_count

        #r_cycles = self.node_compute.get_cycles() #DEPRECATED

        #Create temporary copies of variables to be modified during render
        r_threshold = self.node_compute.get_threshold()
        r_iterations = self.node_compute.get_iterations()

        init_bounds = self.node_compute.get_boundaries()
        r_offset = [
            (init_bounds[1] - init_bounds[0]) /2,
            (init_bounds[3] - init_bounds[2]) /2,
        ]
        r_center = [
            init_bounds[0] + r_offset[0],
            init_bounds[2] + r_offset[1],
        ]

        #Get deltas
        delta_iterations = deltas.get('iterations', 0)
        delta_threshold = deltas.get('threshold', 0)
        delta_x = deltas.get('x', 0)
        delta_y = deltas.get('y', 0)
        delta_scale = deltas.get('scale', 1)

        #RENDER
        for i in range(0, r_frame_count):
            print(f"render frame {i}")
            if i > 0:
                #apply scale
                if i > 1:
                    delta_x *= delta_scale
                    delta_y *= delta_scale
                r_offset[0] *= delta_scale
                r_offset[1] *= delta_scale
                #change variables each iteration after initial
                r_threshold += delta_threshold
                r_iterations += delta_iterations
                if r_iterations < 2:    #enforce lower limit
                    r_iterations = 2
                r_center[0] += delta_x
                r_center[1] += delta_y
            #apply variable changes
            self.node_compute.set_iterations(r_iterations)
            self.node_compute.set_threshold(r_threshold)
            self.node_compute.set_boundaries([
                r_center[0] - r_offset[0], 
                r_center[0] + r_offset[0], 
                r_center[1] - r_offset[1], 
                r_center[1] + r_offset[1]
            ])

            #render current frame
            self.render_frame(i, r_iterations, r_output_dir, degree)

    def render_steps_preview(self, frame_count: int, deltas) -> None: #get state of last frame and render that
        r_frame_count = frame_count

        #r_cycles = self.node_compute.get_cycles() #DEPRECATED

        #Create temporary copies of variables to be modified during render
        r_threshold = self.node_compute.get_threshold()
        r_iterations = self.node_compute.get_iterations()

        init_bounds = self.node_compute.get_boundaries()
        r_offset = [
            (init_bounds[1] - init_bounds[0]) /2,
            (init_bounds[3] - init_bounds[2]) /2,
        ]
        r_center = [
            init_bounds[0] + r_offset[0],
            init_bounds[2] + r_offset[1],
        ]

        #Get deltas
        delta_iterations = deltas.get('iterations', 0)
        delta_threshold = deltas.get('threshold', 0)
        delta_x = deltas.get('x', 0)
        delta_y = deltas.get('y', 0)
        delta_scale = deltas.get('scale', 1)

        #step-by-step
        for i in range(0, r_frame_count):
            if i > 0:
                #apply scale
                if i > 1:
                    delta_x *= delta_scale
                    delta_y *= delta_scale
                r_offset[0] *= delta_scale
                r_offset[1] *= delta_scale
                #change variables each iteration after initial
                r_threshold += delta_threshold
                r_iterations += delta_iterations
                if r_iterations < 2:    #enforce lower limit
                    r_iterations = 2
                r_center[0] += delta_x
                r_center[1] += delta_y

        #apply variable changes
        self.node_compute.set_resolution([100, 100])   #we render at a set 100x100 pixels
        self.node_compute.set_iterations(r_iterations if r_iterations < 50 else int(r_iterations/2))   #iterations get halved
        self.node_compute.set_threshold(r_threshold)
        self.node_compute.set_boundaries([
            r_center[0] - r_offset[0], 
            r_center[0] + r_offset[0], 
            r_center[1] - r_offset[1], 
            r_center[1] + r_offset[1]
        ])


    def render_interpolated(self, output_dir: str, frame_count: int, params, degree=2) -> None:
        #unchanging variables
        #self.active_renderer.set_resolution(self.get_current_height(), self.get_current_width())
        #self.active_renderer.set_threshold(self.get_current_threshold())
        #self.active_renderer.set_iterations(self.get_current_iterations())

        r_output_dir = output_dir
        r_frame_count = frame_count

        r_iterations = self.node_compute.get_iterations()
        #r_cycles = self.node_compute.get_cycles() #DEPRECATED

        mode_scale = params.get('mode_scale', 1) # -1: not applied, 0: linear, 1: exponential 
        mode_trans = params.get('mode_trans', 1) # -1: not applied, 0: linear, 1: scaleDependant

        #start frame (0)
        init_bounds = self.node_compute.get_boundaries()
        start_offset = [
            (init_bounds[1] - init_bounds[0]) /2,
            (init_bounds[3] - init_bounds[2]) /2,
        ]
        start_center = [
            init_bounds[0] + start_offset[0],
            init_bounds[2] + start_offset[1],
        ]

        #end frame (n-1)
        final_bounds = params.get('final_bounds', [-2, 2, -2, 2])
        end_offset = [
            (final_bounds[1] - final_bounds[0]) /2,
            (final_bounds[3] - final_bounds[2]) /2,
        ]
        end_center = [
            final_bounds[0] + end_offset[0],
            final_bounds[2] + end_offset[1],
        ]

        #setup scale deltas
        if end_offset[0] == start_offset[0]: #no scaling change
            mode_scale = -1

        match mode_scale:
            case 0: #linear
                scale_delta_full = end_offset[0] / start_offset[0]
                scale_delta_step = (end_offset[0] - start_offset[0])/(r_frame_count-1) #change to be applied additively
            case 1: #exponential
                scale_delta_full = end_offset[0] / start_offset[0]
                scale_delta_step = pow(scale_delta_full, 1/(r_frame_count-1)) #frame = base * (delta^i)
            case _: #not applied
                #set translation mode from scaleDependant to linear
                if mode_trans == 1:
                    mode_trans = 0
                #default scale values
                scale_delta_full = 1
                scale_delta_step = 1


        #setup translation deltas
        trans_delta_full = [end_center[0] - start_center[0], end_center[1] - start_center[1]]
        trans_delta_step = [trans_delta_full[0]/(r_frame_count-1), trans_delta_full[1]/(r_frame_count-1)]


        #RENDER
        for i in range(0, r_frame_count):
            print(f"render frame {i}")
            #Again: we assume the setup of frame zero has already happened so the values are already set
            #if i == 0:
            #    #render start-frame as written
            #    #self.active_renderer.set_coordinates(start_center_x - start_offset_x, start_center_x + start_offset_x, start_center_y - start_offset_y, start_center_y + start_offset_y)
            #elif i == frame_count-1:
            #    #render end-frame as written
            #    self.active_renderer.set_coordinates(end_center_x - end_offset_x, end_center_x + end_offset_x, end_center_y - end_offset_y, end_center_y + end_offset_y)
            #else:
            if i > 0:
                #interpolate in-between frames
                frame_scale = self.get_frame_scale(i, start_offset[0], scale_delta_step, mode_scale)
                frame_trans = self.get_frame_translation(i, trans_delta_full, trans_delta_step, mode_trans, self.normalize_value(frame_scale, scale_delta_full, 1))
                self.node_compute.set_boundaries([
                    (start_center[0] + frame_trans[0]) - start_offset[0] * frame_scale,
                    (start_center[0] + frame_trans[0]) + start_offset[0] * frame_scale,
                    (start_center[1] + frame_trans[1]) - start_offset[1] * frame_scale,
                    (start_center[1] + frame_trans[1]) + start_offset[1] * frame_scale
                ])
            self.render_frame(i, r_iterations, r_output_dir, degree)

    def get_frame_scale(self, step: int, start_offset_x: float, scale_delta_step: float, mode_scale: int) -> float:
        match mode_scale:
            case 0: #linear
                ## get target offset from linear
                #target_offset_x = start_offset_x + scale_delta_step * step
                ## convert linear to multiplier
                #target_multiplier = target_offset_x/start_offset_x #start * x = target -> x = target/start
                #return target_multiplier
                return (start_offset_x + scale_delta_step * step)/start_offset_x
            case 1: #exponential
                return pow(scale_delta_step, step) #frame = base * (delta^i)
            case _: #not applied
                return 1

    def normalize_value(self, val: float, upper_bound, lower_bound) -> float:
        try:
            return (val - lower_bound) / (upper_bound - lower_bound)
        except ZeroDivisionError:
            return 1

    def get_frame_translation(self, step: int, trans_delta_full: [float, float], trans_delta_step: [float, float], mode_trans: int, frame_scale_normalized: float) -> [float, float]:
        match mode_trans:
            case 0: #linear
                return [
                    trans_delta_step[0] * step, 
                    trans_delta_step[1] * step
                ]
            case 1: #match scale
                return [
                    trans_delta_full[0] * frame_scale_normalized, 
                    trans_delta_full[1] * frame_scale_normalized
                ]
            case _: #not applied
                return [0, 0]

    def render_frame(self, i: int, iterations: float, output_dir: str, degree=2) -> None:
        fractal = self.node_compute.recompute(degree)
        fractal = self.node_colorize.modify_fractal(fractal, self.node_compute.get_iterations()) #compute and colorize no longer support self.node_compute.get_cycles()

        self.node_colorize.save_image(
            f"{output_dir}/{i}.png",
            fractal, 
            iterations=iterations, 
            vmin=1
        )


def create_video(input_dir: str, output_path: str = "output/julia_animation.mp4", fps: int = 10) -> None:
    """Create an MP4 video from PNG frames in input_dir."""
    images = [img for img in os.listdir(input_dir) if img.lower().endswith(".png")]
    if not images:
        print("⚠️ No PNG files found in", input_dir)
        return

    images = natsorted(images)
    first_frame = cv2.imread(os.path.join(input_dir, images[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img in images:
        frame = cv2.imread(os.path.join(input_dir, img))
        video.write(frame)

    video.release()
    print(f"🎞 Julia video saved to {output_path}")