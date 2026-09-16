cmake_minimum_required(VERSION 3.10)
project(lambdaUnoPruebaMarkers)
 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
 
find_package(OpenCV REQUIRED)
 
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    /opt/homebrew/include
    /usr/local/include
)
 
link_directories(
    /opt/homebrew/lib
    /usr/local/lib
)
add_executable(lambdaUnoPruebaMarkers main.cpp vis.cpp) 

target_link_libraries(lambdaUnoPruebaMarkers
    ${OpenCV_LIBS}
)