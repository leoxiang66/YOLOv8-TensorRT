#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check for CUDA
check_cuda() {
    echo "Checking for CUDA installation..."
    if [ -d "/usr/local/cuda" ]; then
        echo "CUDA is installed at /usr/local/cuda"
    else
        echo "CUDA is not installed."
    fi
    echo ""
}

# Check for nvcc
check_nvcc() {
    echo "Checking for nvcc..."
    if command_exists nvcc; then
        nvcc_version=$(nvcc --version | grep release)
        echo "nvcc is installed: $nvcc_version"
    else
        echo "nvcc is not installed."
    fi
    echo ""
}

# Check for cuDNN
check_cudnn() {
    echo "Checking for cuDNN..."
    if [ -f "/usr/include/cudnn.h" ] || [ -f "/usr/local/cuda/include/cudnn.h" ]; then
        cudnn_version=$(grep CUDNN_MAJOR -A 2 /usr/include/cudnn.h 2>/dev/null | grep CUDNN_VERSION | awk '{print $2}' | sed 's/,/./g' | sed 's/"//g')
        echo "cuDNN is installed."
    else
        echo "cuDNN is not installed."
    fi
    echo ""
}

# Check for nvidia-smi
check_nvidia_smi() {
    echo "Checking for nvidia-smi..."
    if command_exists nvidia-smi; then
        nvidia_smi_output=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits)
        echo "nvidia-smi is installed."
        echo "GPU details:"
        echo "$nvidia_smi_output"
    else
        echo "nvidia-smi is not installed."
    fi
    echo ""
}

# Check for TensorRT
check_tensorrt() {
    echo "Checking for TensorRT..."
    if [ -d "/usr/local/TensorRT" ]; then
        tensorrt_version=$(cat /usr/local/TensorRT/include/NvInferVersion.h | grep "#define NV_TENSORRT_MAJOR" | awk '{print $3}')
        echo "TensorRT is installed at /usr/local/TensorRT"
        echo "TensorRT version: $tensorrt_version"
    else
        echo "TensorRT is not installed."
    fi
    echo ""
}

# Check for OpenCV
check_opencv() {
    echo "Checking for OpenCV..."
    if pkg-config --exists opencv4; then
        opencv_version=$(pkg-config --modversion opencv4)
        echo "OpenCV is installed: Version $opencv_version"
    else
        echo "OpenCV is not installed."
    fi
    echo ""
}

# Run all checks
check_nvidia_smi
check_cuda
check_nvcc
check_cudnn
check_tensorrt
check_opencv

echo "All checks completed."