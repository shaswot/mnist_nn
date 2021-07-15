cmake_minimum_required(VERSION 3.10)

# set the project name
project(mmultiply
    VERSION 1.0
    DESCRIPTION "A brief CMake experiment"
    LANGUAGES CXX)
    
# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
  
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

# add the executable
add_executable(mmultiply
    src/mmultiply.cpp
    include/mnist_loader.cc
    include/mnist_loader.h
    )

# https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
# warning options
target_compile_options(mmultiply PRIVATE -Wall -Wextra -Wpedantic)