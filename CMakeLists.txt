cmake_minimum_required(VERSION 2.8.12)

project(getPhiArmDynamics)

find_package(DART 6 REQUIRED COMPONENTS utils-urdf)

add_compile_options(-std=c++11)

include_directories(${DART_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS})

file(GLOB srcs "*.cpp" "*.hpp")
add_executable(${PROJECT_NAME} ${srcs})

target_link_libraries(${PROJECT_NAME} ${DART_LIBRARIES} nlopt)
