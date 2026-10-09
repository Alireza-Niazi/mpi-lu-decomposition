cmake_minimum_required(VERSION 3.10)
project(MPI_LU_Decomposition)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

# Find MPI
find_package(MPI REQUIRED)

# Add executable
add_executable(mpi_lu src/mpi_lu.cpp)

# Link MPI libraries
target_link_libraries(mpi_lu PUBLIC MPI::MPI_C)
target_include_directories(mpi_lu PUBLIC ${MPI_C_INCLUDE_PATH})

# Compiler flags
if(UNIX)
    target_compile_options(mpi_lu PRIVATE -Wall -Wextra -O2)
endif()
