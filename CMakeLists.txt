cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")

set(DEMO face_recognition_demo)
set(CLIENT face_recognition_client)
set(SERVER face_recognition_server)

find_package(TensorflowCC REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

SET(faceRecognitionFiles
    faceAntiSpoofing.cpp
    faceDetection.cpp
    faceRecognition.cpp
)

add_executable(${DEMO} main.cpp ${faceRecognitionFiles})
add_executable(${CLIENT} cppSocket.cpp cameraClient.cpp ${faceRecognitionFiles})
add_executable(${SERVER} cppSocket.cpp demoServer.cpp ${faceRecognitionFiles})

target_link_libraries(${DEMO} TensorflowCC::Shared ${OpenCV_LIBS} tensorflowlite ${NCNN_LIB} ncnn jsoncpp)
target_link_libraries(${CLIENT} TensorflowCC::Shared ${OpenCV_LIBS} tensorflowlite ${NCNN_LIB} ncnn jsoncpp)
target_link_libraries(${SERVER} TensorflowCC::Shared ${OpenCV_LIBS} tensorflowlite ${NCNN_LIB} ncnn jsoncpp)

if (GPU_DELEGATE)
    add_definitions(-DGPU_DELEGATE)
    target_link_libraries(${DEMO} tensorflowlite_gpu_delegate m EGL GLESv2)
    target_link_libraries(${CLIENT} tensorflowlite_gpu_delegate m EGL GLESv2)
    target_link_libraries(${SERVER} tensorflowlite_gpu_delegate m EGL GLESv2)
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/face_database DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
