#!/bin/bash

# generate a fractal (numpy array) with some arbitrary values for all available params (low resolution/iterations for quick response)
# save the generated fractal as a png with some arbitrary coloring params

# give this file execute privileges if you get permission denied:
#  chmod +x generate_fractal.bash

# starting the computation & coloring requires your console environment to be sourced correctly

# navigate to main folder
cd ..
# compute a fractal & save it as a numpy array
python compute.py --output example_bashs/test_bash_compute --use_complex 0 --in_place 1 --use_mask 1 --resolution 100 --xmin -2.0 --xmax 2.0 --ymin -2.0 --ymax 2.0 --iterations 10 --threshold 2.0
# color & save the fractal as a png
python colorize.py --input example_bashs/test_bash_compute --output example_bashs/test_bash_colorize --coloring Sqrt --palette Blues --iterations 10 --cycles 1.0
# go back to current folder
cd example_bashs