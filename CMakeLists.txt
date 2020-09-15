cmake_minimum_required(VERSION 2.6)

project(face-recognition-tensorrt)

set(TARGET_NAME "face-recognition-tensorrt")
set(CMAKE_CXX_STANDARD 17)

# Find necessary libraries
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(ncnn REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${OpenCV_INCLUDE_DIRS}
                    ${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

file(GLOB CUDA_SOURCES "*.cu")

cuda_add_library(yolov5_plugin SHARED ${CUDA_SOURCES})

# TensorRT libs usage
list(APPEND TensorRT_PLUGINS "nvinfer")
list(APPEND TensorRT_PLUGINS "nvinfer_plugin")
list(APPEND TensorRT_PLUGINS "nvparsers")

foreach(libName ${TensorRT_PLUGINS})
  message(STATUS "libName: ${libName}")
  find_library(TensorRT_${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIXES lib)
  list(APPEND TensorRT_PLUGIN_LIBS "${TensorRT_${libName}_lib}")
endforeach()

file(GLOB SOURCES "*.cpp")

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME}
                      ${OpenCV_LIBS}
                      ${TensorRT_PLUGIN_LIBS}
                      ${CUDA_LIBRARIES}
                      ${CUDA_CUBLAS_LIBRARIES}
                      ${NCNN_LIB}
                      yolov5_plugin
                      ncnn
                      jsoncpp)

# Copy models to built folder
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
