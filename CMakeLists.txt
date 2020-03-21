cmake_minimum_required(VERSION 3.0.0)
project(DroneStateEstimation VERSION 0.1.0)

#[[
include(CTest)
enable_testing()
]]

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(DroneStateEstimation src/main.cpp src/vehicles.cpp)
target_link_libraries (DroneStateEstimation Eigen3::Eigen)

target_include_directories(DroneStateEstimation
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

#[[
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
]]

target_compile_features(DroneStateEstimation PRIVATE cxx_std_17)
#[[
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
]]