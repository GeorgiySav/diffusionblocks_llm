cmake_minimum_required(VERSION 3.20)
project(llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# nn-library is a sibling checkout, not a subdirectory of this repo -- it stays
# a normal, independent git clone. Editing and committing it is just `cd` into
# NN_LIBRARY_DIR and using git as usual; there's nothing to sync back into this
# repo. Override with -DNN_LIBRARY_DIR=... if the checkout lives elsewhere
# (e.g. on another machine or CI).
#
# Declared blank rather than defaulted directly in the CACHE call: some
# configure paths (IDE tooling, presets) pre-seed a cache entry with an empty
# value before this line runs, and set(... CACHE ...) without FORCE never
# overwrites an existing entry -- silently passing add_subdirectory an empty
# path. Falling back explicitly here recomputes the default whenever the
# variable is blank, no matter how it got that way
set(NN_LIBRARY_DIR "" CACHE PATH "Path to the nn library checkout")
if(NOT NN_LIBRARY_DIR)
  set(NN_LIBRARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../nn library")
endif()

# This project builds its own executable(s); it doesn't need nn's example or
# benchmark targets. Must be set before add_subdirectory -- nn's own option()
# calls do not override an already-cached value.
set(NN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NN_BUILD_BENCH OFF CACHE BOOL "" FORCE)

add_subdirectory("${NN_LIBRARY_DIR}" "${CMAKE_BINARY_DIR}/nn-library")

# Support sources live in subdirectories of src/ (e.g. src/tokenizer/) and
# have no main() of their own -- every executable below links all of them.
# Entry points are the top-level .cpp files directly under src/, each with
# its own main(); one executable is created per entry point, named after the
# file. CONFIGURE_DEPENDS makes both globs re-run at build time so new files
# are picked up without re-running cmake by hand.
file(GLOB_RECURSE LLM_SUPPORT_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*/*.cpp"
)
file(GLOB LLM_ENTRY_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

foreach(entry_source ${LLM_ENTRY_SOURCES})
  get_filename_component(entry_name ${entry_source} NAME_WE)
  add_executable(${entry_name} ${entry_source} ${LLM_SUPPORT_SOURCES})
  target_link_libraries(${entry_name} PRIVATE nn)
  # A quoted #include only reliably reaches its own directory on standard
  # (non-MSVC) toolchains; without src/ on the path, a subdirectory header
  # (e.g. src/llm/trainer.h) that reaches for a sibling of src/ (e.g.
  # src/train_config.h) would resolve only via clang-cl/MSVC's own extra,
  # non-portable "search every ancestor directory" behaviour.
  target_include_directories(${entry_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
endforeach()

# Tests get their own executable, built the same way nn-library's own
# test_all is: one binary over every tests/*.cpp, reusing nn's header-only
# NN_TEST/NN_CHECK harness (private to nn-library, not installed, so it's
# reached via NN_LIBRARY_DIR rather than the public include/ path).
enable_testing()
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
if(TEST_SOURCES)
  add_executable(llm_tests ${TEST_SOURCES})
  target_link_libraries(llm_tests PRIVATE nn)
  target_include_directories(llm_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${NN_LIBRARY_DIR}/tests")
  add_test(NAME unit COMMAND llm_tests)
endif()
