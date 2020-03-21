cmake_minimum_required(VERSION 3.0.0)
project(DroneStateEstimation VERSION 0.1.0)


# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download"
)
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download"
)

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This adds the following targets:
# gtest, gtest_main, gmock and gmock_main
add_subdirectory("${CMAKE_BINARY_DIR}/googletest-src"
                 "${CMAKE_BINARY_DIR}/googletest-build"
)

# The gtest/gmock targets carry header search path dependencies
# automatically when using CMake 2.8.11 or later. Otherwise we
# have to add them here ourselves.
if(CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include"
                        "${gmock_SOURCE_DIR}/include"
    )
endif()


#[[
include(CTest)
enable_testing()
]]

#find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_subdirectory(src)

add_executable(DroneStateEstimation main.cpp)
# target_link_libraries (DroneStateEstimation Eigen3::Eigen dronelib)
target_link_libraries (DroneStateEstimation dronelib)


#[[
add_executable(UnitTests "")
target_link_libraries (UnitTests gtest_main)

target_sources(UnitTests
    PRIVATE
        unit_tests.cpp 
        src/sample1.cpp
)
# add_test(NAME example_test COMMAND example)
]]

target_include_directories(DroneStateEstimation
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
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