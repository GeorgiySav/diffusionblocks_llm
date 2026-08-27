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

add_executable(llm src/main.cpp)
target_link_libraries(llm PRIVATE nn)
