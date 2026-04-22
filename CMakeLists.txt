cmake_minimum_required(VERSION 3.16)
project(EigenEngine LANGUAGES CXX)

# =====================
# C++ Standard
# =====================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =====================
# Executable
# =====================
add_executable(eigen_engine
    main.cpp
    matrix.cpp
    solver.cpp
    pca.cpp
    benchmark.cpp
    cli.cpp
)

# =====================
# Include directories
# =====================
target_include_directories(eigen_engine PRIVATE src)

# =====================
# Compiler options (target-specific)
# =====================
target_compile_options(eigen_engine PRIVATE
-O3
-Wall
-Wextra
-Wpedantic
)

# =====================
# Build type (default: Release)
# =====================
if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif()

# =====================
# Optional: Native optimization (only if enabled)
# =====================
option(ENABLE_NATIVE "Enable -march=native" OFF)

if(ENABLE_NATIVE)
target_compile_options(eigen_engine PRIVATE -march=native)
endif()

# =====================
# Threads
# =====================
find_package(Threads REQUIRED)
target_link_libraries(eigen_engine PRIVATE Threads::Threads)
