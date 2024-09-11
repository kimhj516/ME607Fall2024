cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

project(ME607Fall2024)

#################################################################################################################
# add trilinos library
find_package(Trilinos)

MESSAGE("\nFound Trilinos!  Here are the details: ")
MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
MESSAGE("   Trilinos_CXX_COMPILER = ${Trilinos_CXX_COMPILER}")
MESSAGE("   Trilinos_C_COMPILER = ${Trilinos_C_COMPILER}")
MESSAGE("   Trilinos_Fortran_COMPILER = ${Trilinos_Fortran_COMPILER}")
MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
MESSAGE("   Trilinos_C_COMPILER_FLAGS = ${Trilinos_C_COMPILER_FLAGS}")
MESSAGE("   Trilinos_Fortran_COMPILER_FLAGS = ${Trilinos_Fortran_COMPILER_FLAGS}")
MESSAGE("   Trilinos_LINKER = ${Trilinos_LINKER}")
MESSAGE("   Trilinos_EXTRA_LD_FLAGS = ${Trilinos_EXTRA_LD_FLAGS}")
MESSAGE("   Trilinos_AR = ${Trilinos_AR}")
MESSAGE("End of Trilinos details\n")

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -pedantic -std=c++11")
set(CMAKE_C_FLAGS "-Wall -Wextra -Wconversion -pedantic -std=gnu11")
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -Wconversion -pedantic -std=c++11 -g -O0")
set(CMAKE_C_FLAGS_DEBUG "-Wall -Wextra -Wconversion -pedantic -std=gnu11 -g -O0")

find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER "/usr/bin/mpicxx")
set(CMAKE_C_COMPILER "/usr/bin/mpicc")
message (STATUS "MPI_VERSION: ${MPI_VERSION}")
message(STATUS "MPI_INCLUDE_DIRS = ${MPI_INCLUDE_DIRS}")
message(STATUS "MPI_LIBRARIES = ${MPI_LIBRARIES}")

list(APPEND CMAKE_PREFIX_PATH "$HOM/opt/homebrew/opt/libomp/")
find_package(OpenMP REQUIRED)
message (STATUS "OpenMP_VERSION: ${OpenMP_VERSION}")
message(STATUS "OpenMP_INCLUDE_DIRS = ${OpenMP_INCLUDE_DIRS}")
message(STATUS "OpenMP_LIBRARIES = ${OpenMP_LIBRARIES}")

find_package(Boost 1.66 REQUIRED COMPONENTS filesystem log system thread filesystem date_time program_options unit_test_framework regex program_options)
message(STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}")

message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_C_FLAGS = ${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_C_FLAGS_DEBUG = ${CMAKE_C_FLAGS_DEBUG}")

enable_testing()
add_subdirectory(practice_01)
