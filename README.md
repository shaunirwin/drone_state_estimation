# README

## Overview

This project is for creating a simulation of moving vehicles or objects and using various state estimation techniques to estiamte the vehicle states.

Vehicles currently supported:
* Drone with 3D position and velocity states that flies in a random walk motion

## Setup

~~~
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=<path to the cmake build folder of your Eigen installation>
make
~~~

## Unit tests

To run unit tests:
~~~
./UnitTests
~~~