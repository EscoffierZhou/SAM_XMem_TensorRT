cmake_minimum_required(VERSION 3.10)
project(SAM_TensorRT_Test)

set(CMAKE_CXX_STANDARD 17)

# 1. 找 OpenCV (用我们自己编译的)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 2. 找 TensorRT 和 CUDA
# 因为是 apt 安装的，通常在标准路径，或者需要指定
find_path(TRT_INCLUDE_DIR NvInfer.h HINTS /usr/include /usr/src/tensorrt/include)
find_library(TRT_LIB nvinfer HINTS /usr/lib /usr/lib/x86_64-linux-gnu)

find_package(CUDAToolkit REQUIRED)

include_directories(${TRT_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS} src)

# 3. 编译
add_executable(sam_test src/main.cpp src/SamEngine.cpp src/SamDecoder.cpp)

# 4. 链接
target_link_libraries(sam_test
        ${OpenCV_LIBS}
        ${TRT_LIB}
        CUDA::cudart
)