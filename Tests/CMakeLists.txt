# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

find_package(GTest CONFIG REQUIRED)

add_executable("${PROJECT_NAME}_test"
  HelloTest.cxx
  EbrTest.cxx
)
target_link_libraries(
  "${PROJECT_NAME}_test"
  PRIVATE
    GTest::gtest
    GTest::gtest_main
    ${PROJECT_NAME}::${PROJECT_NAME}
)
target_include_directories(
  "${PROJECT_NAME}_test"
  PRIVATE
    "${CMAKE_SOURCE_DIR}/Source"
)

# Enable address sanitizer for tests when supported, but skip if TSan is already active.
set(_conflux_enable_test_asan ON)
foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_MINSIZEREL)
  if(DEFINED ${flag_var} AND "${${flag_var}}" MATCHES "sanitize=thread")
    set(_conflux_enable_test_asan OFF)
  endif()
endforeach()

if(_conflux_enable_test_asan AND (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU"))
  target_compile_options(
    "${PROJECT_NAME}_test"
    PRIVATE
      -fsanitize=address
      -fno-omit-frame-pointer
  )
  target_link_options(
    "${PROJECT_NAME}_test"
    PRIVATE
      -fsanitize=address
  )
elseif(_conflux_enable_test_asan AND MSVC)
  # MSVC address sanitizer is available in recent versions.
  target_compile_options("${PROJECT_NAME}_test" PRIVATE /fsanitize=address)
  target_link_options("${PROJECT_NAME}_test" PRIVATE /fsanitize=address)
endif()

include(GoogleTest)
gtest_discover_tests("${PROJECT_NAME}_test" DISCOVERY_TIMEOUT 60)
