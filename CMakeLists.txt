cmake_minimum_required(VERSION 4.0.1)

project(
  Conflux
  VERSION 0.1.0
  DESCRIPTION "Lock-free queues and epoch-based reclamation utilities"
  LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CONFLUX_BUILD_TESTS "Build Conflux unit tests" OFF)
option(CONFLUX_BUILD_BENCHMARKS "Build Conflux benchmarks" OFF)

add_subdirectory(Source)

if(CONFLUX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(Tests)
endif()

if(CONFLUX_BUILD_BENCHMARKS)
  add_subdirectory(Benchs)
endif()

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/confluxConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/conflux-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/confluxConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(
  EXPORT confluxTargets
  NAMESPACE ${PROJECT_NAME}::
  FILE confluxTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/confluxConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/confluxConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
