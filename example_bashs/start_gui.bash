#!/bin/bash

# start the GUI with some arbitrary values for all available params (low resolution/iterations for quick response)

# give this file execute privileges if you get permission denied:
#  chmod +x start_gui.bash

# starting the gui requires your console environment to be sourced correctly

# navigate to main folder
cd ..
# start gui
python gui.py --use_complex 0 --in_place 1 --use_mask 1 --resolution 100 --xmin -2.0 --xmax 2.0 --ymin -2.0 --ymax 2.0 --iterations 10 --threshold 2.0 --cycles 1.0 --coloring Sqrt --palette Blues
# go back to current folder
cd example_bashs