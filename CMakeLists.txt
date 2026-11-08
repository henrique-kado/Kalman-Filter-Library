cmake_minimum_required(VERSION 3.16)
project(kalman VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ---------------------------------------------------------------------------
# Library (header-only)
# ---------------------------------------------------------------------------
add_library(kalman INTERFACE)
add_library(kalman::kalman ALIAS kalman)

# Headers live in include/ while building and in <prefix>/include once installed.
target_include_directories(kalman INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(kalman INTERFACE cxx_std_17)
target_link_libraries(kalman INTERFACE Eigen3::Eigen)

# ---------------------------------------------------------------------------
# Options
#
# Tests, examples and install rules are on only when this is the main project,
# not when another project pulls the library in with add_subdirectory or
# FetchContent.
# Override with -DKALMAN_BUILD_TESTS=ON/OFF, -DKALMAN_BUILD_EXAMPLES=ON/OFF
# and -DKALMAN_INSTALL=ON/OFF.
# ---------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(KALMAN_IS_MAIN_PROJECT ON)
else()
    set(KALMAN_IS_MAIN_PROJECT OFF)
endif()
option(KALMAN_BUILD_TESTS "Build the kalman unit tests" ${KALMAN_IS_MAIN_PROJECT})
option(KALMAN_BUILD_EXAMPLES "Build the kalman examples" ${KALMAN_IS_MAIN_PROJECT})
option(KALMAN_INSTALL "Generate the kalman install rules" ${KALMAN_IS_MAIN_PROJECT})

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(KALMAN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------
# Install: lets other projects use find_package(kalman)
#
#   <prefix>/include/kalman/*.hpp
#   <prefix>/share/kalman/cmake/kalmanConfig.cmake
#   <prefix>/share/kalman/cmake/kalmanConfigVersion.cmake
#   <prefix>/share/kalman/cmake/kalmanTargets.cmake
# ---------------------------------------------------------------------------
if(KALMAN_INSTALL)
    include(CMakePackageConfigHelpers)

    set(KALMAN_CMAKE_DIR ${CMAKE_INSTALL_DATADIR}/kalman/cmake)

    install(TARGETS kalman EXPORT kalmanTargets)
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(EXPORT kalmanTargets
        FILE kalmanTargets.cmake
        NAMESPACE kalman::
        DESTINATION ${KALMAN_CMAKE_DIR})

    configure_package_config_file(
        ${PROJECT_SOURCE_DIR}/cmake/kalmanConfig.cmake.in
        ${PROJECT_BINARY_DIR}/kalmanConfig.cmake
        INSTALL_DESTINATION ${KALMAN_CMAKE_DIR})

    # Header-only: the same package works for 32- and 64-bit consumers.
    write_basic_package_version_file(
        ${PROJECT_BINARY_DIR}/kalmanConfigVersion.cmake
        COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT)

    install(FILES
        ${PROJECT_BINARY_DIR}/kalmanConfig.cmake
        ${PROJECT_BINARY_DIR}/kalmanConfigVersion.cmake
        DESTINATION ${KALMAN_CMAKE_DIR})
endif()
