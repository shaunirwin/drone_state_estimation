# README

## Overview

This project is for creating a simulation of moving vehicles or objects and using various state estimation techniques to estiamte the vehicle states.

Vehicles and state estimation algorithms are implemented in C++ and simulation data is written to binary files for reading in and visualising with Python and Matplotlib.

Vehicles currently supported:
* Drone with 3D position and velocity states that flies in a random walk motion

See the [documentation](docs/docs.tex) for more information on the vehicle models.

## Setup

~~~
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=<path to the cmake build folder of your Eigen installation>
make
~~~

Compiling Latex documentation:
~~~
cd docs
pdflatex docs.tex
~~~

## Run simulation

~~~
./DroneStateEstimation
~~~

## Unit tests

To run unit tests:
~~~
./UnitTests
~~~