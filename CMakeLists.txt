cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(APP "app")
project(${APP})

set(CMAKE_CXX_STANDARD 20)
set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS}")
# set(CMAKE_PREFIX_PATH /usr/local/libtorch)
# set(CMAKE_PREFIX_PATH /usr)

# OpenCV
find_package(OpenCV REQUIRED)
set(LIB ${LIB} ${OpenCV_LIBRARIES})
set(INC ${INC} ${OpenCV_INCLUDE_DIRS})

# Pytorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
message(STATUS "CMAKE_CXX_FLAGS:     ${CMAKE_CXX_FLAGS}")
message(STATUS "TORCH_INCLUDE_DIRS:  ${TORCH_INCLUDE_DIRS}")
message(STATUS "TORCH_LIBRARIES:     ${TORCH_LIBRARIES}")
set(LIB ${LIB} ${TORCH_LIBRARIES})
set(INC ${INC} ${TORCH_INCLUDE_DIRS})

# AlphaPose.cpp
set(SRC ${SRC} AlphaPose.cpp)

# Main.cpp
set(SRC ${SRC} main.cpp)

# Target
include_directories(${INC})
add_executable(${APP} ${SRC})
target_link_libraries(${APP} ${LIB})
