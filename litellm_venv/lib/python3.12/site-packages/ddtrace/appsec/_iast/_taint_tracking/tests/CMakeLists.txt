include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip)
FetchContent_MakeAvailable(googletest)

enable_testing()

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${PYBIND11_INCLUDE_DIRS})

file(GLOB TEST_SOURCES "*.cpp" "*.hpp")
add_executable(native_tests ${TEST_SOURCES} ${SOURCE_FILES} ${HEADER_FILES})

set(NATIVE_TEST_LIBRARIES gtest gtest_main ${PYTHON_LIBRARIES} pybind11::module)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND NATIVE_TEST_LIBRARIES absl::node_hash_map)
endif()

if(NOT MSVC AND NATIVE_TEST_COVERAGE)
    target_compile_options(native_tests PRIVATE -ggdb --coverage)
    list(APPEND NATIVE_TEST_LIBRARIES gcov)
endif()

target_link_libraries(native_tests ${NATIVE_TEST_LIBRARIES})

# Discover tests
include(GoogleTest)
gtest_discover_tests(native_tests)

add_custom_target(
    test_valgrind
    COMMAND valgrind --leak-check=full --suppressions=../../../../../scripts/iast/valgrind-python.supp
            --show-reachable=yes ${CMAKE_BINARY_DIR}/tests/native_tests --gtest_filter=-TestTimer.*
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_dependencies(test_valgrind native_tests)
