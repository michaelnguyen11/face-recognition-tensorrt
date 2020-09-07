cmake_minimum_required(VERSION 2.6)

project(face-recognition-tensorrt)

set(TARGET_NAME "face-recognition-tensorrt")
set(CMAKE_CXX_STANDARD 11)

# Find necessary libraries
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${OpenCV_INCLUDE_DIRS}
                    ${CUDA_INCLUDE_DIRS})


# TensorRT libs usage
list(APPEND TensorRT_PLUGINS "nvinfer")
list(APPEND TensorRT_PLUGINS "nvinfer_plugin")
list(APPEND TensorRT_PLUGINS "nvparsers")

foreach(libName ${TensorRT_PLUGINS})
  message(STATUS "libName: ${libName}")
  find_library(TensorRT_${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIXES lib)
  list(APPEND TensorRT_PLUGIN_LIBS "${TensorRT_${libName}_lib}")
endforeach()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("Using ARM platform")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("Using X86 platform")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()

file(GLOB SOURCES "*.cpp")

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS} ${TensorRT_PLUGIN_LIBS} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
