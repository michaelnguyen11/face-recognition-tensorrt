cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

set (CMAKE_CXX_STANDARD 17)
set(DEMO face_recognition_demo)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")

file(GLOB SRC "*.cpp")
add_executable(${DEMO} ${SRC})

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

target_link_libraries(${DEMO} ${OpenCV_LIBS} tensorflowlite ${NCNN_LIB} ncnn jsoncpp)

if (GPU_DELEGATE)
    add_definitions(-DGPU_DELEGATE)
    target_link_libraries(${DEMO} tensorflowlite_gpu_delegate m EGL GLESv2)
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/face_database DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
