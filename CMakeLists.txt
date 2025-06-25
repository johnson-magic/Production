cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(Production)
set(CMAKE_BUILD_TYPE Release)

option(SPEED_TEST "SPEED TEST" OFF)
option(CONFORMANCE_TEST "conformance test" OFF)
option(ENCRYPT "encrypt" OFF)
if(SPEED_TEST)
    add_definitions(-DSPEED_TEST)
endif()
if(CONFORMANCE_TEST)
    add_definitions(-DCONFORMANCE_TEST)
endif()
if(ENCRYPT)
    add_definitions(-DENCRYPT)
endif()

# 查找 OpenCV 包
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/build/build/generators/")

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(detector_inference REQUIRED)
find_package(classifier_inference REQUIRED)
find_package(sequencer_inference REQUIRED)

# 设置可执行文件名
set(SOURCES
    src/main.cpp
    src/detector_pipeline.cpp
    src/classifier_pipeline.cpp
    src/sequence_pipeline.cpp
    src/plugin_pipeline.cpp
    src/utils.cpp
)
add_executable(${PROJECT_NAME} ${SOURCES}) # main.cpp 为你的源文件名称
include_directories("./include")

# 链接 OpenCV 库和 ONNX Runtime 库
target_link_libraries(${PROJECT_NAME}
                      opencv::opencv
                      onnxruntime::onnxruntime
                      detector_inference::detector_inference
                      classifier_inference::classifier_inference
                      sequencer_inference::sequencer_inference)

# 设置 C++ 标准
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES)
