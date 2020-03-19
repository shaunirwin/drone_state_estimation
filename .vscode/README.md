# README

## Setup

~~~
sudo apt install build-essential clang cmake

# install GLFW
git clone https://github.com/glfw/glfw.git
cd glfw/
mkdir build
cd build
cmake ..
<install libx11-dev and all other dependencies required to build GLFW>
make
~~~