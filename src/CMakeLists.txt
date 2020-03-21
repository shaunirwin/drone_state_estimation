cmake_minimum_required(VERSION 3.13)

project(dronelib_project)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_library(dronelib STATIC vehicles.cpp sample1.cpp)
# add_library(dronelib STATIC vehicles.cpp)

target_link_libraries (dronelib Eigen3::Eigen)

target_include_directories(dronelib
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

#include_directories(../include)

# install(TARGETS dronelib DESTINATION lib)
# install(FILES test.h DESTINATION include)