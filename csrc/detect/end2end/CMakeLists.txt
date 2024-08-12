cmake_minimum_required(VERSION 3.12)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 89 90)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(yolov8 LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Function)

# CUDA
find_package(CUDA REQUIRED)
print_var(CUDA_LIBRARIES)
print_var(CUDA_INCLUDE_DIRS)
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
print_var(CUDA_LIB_DIR)

# OpenCV
find_package(OpenCV REQUIRED)
print_var(OpenCV_LIBS)
print_var(OpenCV_LIBRARIES)
print_var(OpenCV_INCLUDE_DIRS)

# TensorRT
set(TensorRT_DIR /usr/local/TensorRT/)
set(TensorRT_LIB_DIR ${TensorRT_DIR}/lib/)
set(TensorRT_INCLUDE_DIR ${TensorRT_DIR}/include)
set(TensorRT_LIBRARY ${TensorRT_LIB_DIR}/libnvinfer.so)
set(TensorRT_PLUGIN_LIBRARY ${TensorRT_LIB_DIR}/libnvinfer_plugin.so)

find_package(TensorRT REQUIRED)
set(TensorRT_LIBRARIES ${TensorRT_LIBRARY} ${TensorRT_PLUGIN_LIBRARY})
print_var(TensorRT_LIBRARIES)
print_var(TensorRT_INCLUDE_DIRS)
print_var(TensorRT_LIB_DIR)
if (TensorRT_VERSION_MAJOR GREATER_EQUAL 10)
    message(STATUS "Build with -DTRT_10")
    add_definitions(-DTRT_10)
endif ()

list(APPEND ALL_INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
)

list(APPEND ALL_LIB_DIRS
        ${CUDA_LIB_DIR}
        ${TensorRT_LIB_DIR}
)

print_var(ALL_INCLUDE_DIRS)
print_var(ALL_LIBS)
print_var(ALL_LIB_DIRS)

# Executable for main.cpp
add_executable(
        yolov8_main1
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yolov8.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/common.hpp
)

set_target_properties(yolov8_main1 PROPERTIES SUFFIX ".out")

target_include_directories(
        yolov8_main1
        PUBLIC
        ${ALL_INCLUDE_DIRS}
)

target_link_directories(
        yolov8_main1
        PUBLIC
        ${ALL_LIB_DIRS}
)

target_link_libraries(
        yolov8_main1
        PRIVATE
        ${ALL_LIBS}
)

# Executable for main2.cpp
add_executable(
        yolov8_main2
        ${CMAKE_CURRENT_SOURCE_DIR}/main_textoutput.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yolov8.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/common.hpp
)

set_target_properties(yolov8_main2 PROPERTIES SUFFIX ".out")

target_include_directories(
        yolov8_main2
        PUBLIC
        ${ALL_INCLUDE_DIRS}
)

target_link_directories(
        yolov8_main2
        PUBLIC
        ${ALL_LIB_DIRS}
)

target_link_libraries(
        yolov8_main2
        PRIVATE
        ${ALL_LIBS}
)


# Executable for main_time.cpp
add_executable(
        yolov8_main3
        ${CMAKE_CURRENT_SOURCE_DIR}/main_time.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yolov8.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/common.hpp
)

set_target_properties(yolov8_main3 PROPERTIES SUFFIX ".out")

target_include_directories(
        yolov8_main3
        PUBLIC
        ${ALL_INCLUDE_DIRS}
)

target_link_directories(
        yolov8_main3
        PUBLIC
        ${ALL_LIB_DIRS}
)

target_link_libraries(
        yolov8_main3
        PRIVATE
        ${ALL_LIBS}
)