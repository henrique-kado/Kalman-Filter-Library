cmake_minimum_required(VERSION 3.16)
project(kalman VERSION 0.1.0 LANGUAGES CXX)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(kalman INTERFACE)
add_library(kalman::kalman ALIAS kalman)
target_include_directories(kalman INTERFACE include)
target_compile_features(kalman INTERFACE cxx_std_17)
target_link_libraries(kalman INTERFACE Eigen3::Eigen)

# Tests are built only when this is the main project, not when another
# project pulls the library in with add_subdirectory or FetchContent.
# Override with -DKALMAN_BUILD_TESTS=ON or OFF.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(KALMAN_IS_MAIN_PROJECT ON)
else()
    set(KALMAN_IS_MAIN_PROJECT OFF)
endif()
option(KALMAN_BUILD_TESTS "Build the kalman unit tests" ${KALMAN_IS_MAIN_PROJECT})

if(KALMAN_BUILD_TESTS)
    enable_testing()

    add_executable(test_filter_base tests/test_filter_base.cpp)
    target_link_libraries(test_filter_base PRIVATE kalman::kalman)
    add_test(NAME test_filter_base COMMAND test_filter_base)

    add_executable(test_kalman_filter tests/test_kalman_filter.cpp)
    target_link_libraries(test_kalman_filter PRIVATE kalman::kalman)
    add_test(NAME test_kalman_filter COMMAND test_kalman_filter)

    add_executable(test_extended_kalman_filter tests/test_extended_kalman_filter.cpp)
    target_link_libraries(test_extended_kalman_filter PRIVATE kalman::kalman)
    add_test(NAME test_extended_kalman_filter COMMAND test_extended_kalman_filter)

    add_executable(test_unscented_kalman_filter tests/test_unscented_kalman_filter.cpp)
    target_link_libraries(test_unscented_kalman_filter PRIVATE kalman::kalman)
    add_test(NAME test_unscented_kalman_filter COMMAND test_unscented_kalman_filter)
endif()
