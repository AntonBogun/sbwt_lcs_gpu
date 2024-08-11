# This is the root CMakeLists which calls all other cmake files. It is used by
# the cmake tool to build the Makefile which would then be used to compile our
# program. Inside it, we specify the project name and details, we load all
# files within the cmake folder and then add the src and documentation folder,
# which also contain a CMakeLists.txt file, as subdirectories, so that the
# CMakeLists.txt file within them is also executed (sequentially).  Since we
# will be using CUDA and HIP, we want to use a pretty high version of CMake.

cmake_minimum_required (VERSION 3.18)#needed for CUDA 11.5.0
project(SBWT_LCS_GPU LANGUAGES CXX CUDA)
include(ExternalProject)
enable_language(CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



## Get version number from git tags
find_package(Git)
if(GIT_FOUND)
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
  endif()
  # execute_process(
  #   COMMAND ${GIT_EXECUTABLE} describe --tags
  #   WORKING_DIRECTORY "${local_dir}"
  #   OUTPUT_VARIABLE SBWT_BUILD_VERSION
  #   ERROR_QUIET
  #   OUTPUT_STRIP_TRAILING_WHITESPACE
  # )
  # message(STATUS "SBWT version: ${SBWT_BUILD_VERSION}")
else()
  message(STATUS "GIT not found, version numbering disabled")
  set(SBWT_BUILD_VERSION "unknown")
endif()
# string(TIMESTAMP SBWT_BUILD_TIMESTAMP)

## Set maximum k-mer length
if(NOT MAX_KMER_LENGTH)
  set(MAX_KMER_LENGTH 32)
endif()
if(MAX_KMER_LENGTH GREATER 255)
  message(${MAX_KMER_LENGTH})
  message(FATAL_ERROR "Error: MAX_KMER_LENGTH can be at most 255")
endif()
add_definitions(-DMAX_KMER_LENGTH=${MAX_KMER_LENGTH}) # Define for compiler.
message(STATUS "Maximum k-mer length is: " ${MAX_KMER_LENGTH})
message(STATUS "To change the maximum k-mer length, use the option -DMAX_KMER_LENGTH=n")
message(STATUS "(n can be up to 255)")



# Fix for GCC (see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=90050 )
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CXX_FILESYSTEM_LIBRARIES "stdc++fs")
endif()




# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(cmake/PreventBuildingInCmakeDirectory.cmake)
include(cmake/CompilerWarnings.cmake)
include(cmake/Profiling.cmake)
include(cmake/Options.cmake)
include(cmake/CCache.cmake)
include(cmake/CCache.cmake)
# include(cmake/SetHipTargetDevice.cmake)


find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)


# git submodule add https://github.com/iosfwd/sdsl-lite.git sdsl-lite
## Add local dependencies as targets
add_subdirectory(${PROJECT_SOURCE_DIR}/sdsl-lite
  ${PROJECT_BINARY_DIR}/external/sdsl-lite/build)#need to build otherwise linking fails

# add_subdirectory(${PROJECT_SOURCE_DIR}/KMC
#   ${PROJECT_BINARY_DIR}/external/KMC/build)

# Our sources, excluding the file with the main function
set(SBWT_SOURCES
  # src/globals.cpp
  # src/suffix_group_optimization.cpp
  # src/EM_sort/Block.cpp
  # src/EM_sort/EM_sort.cpp
  # src/SeqIO.cpp src/run_kmc.cpp
  # src/kmc_construct_helper_classes.cpp
  # main.cpp
  )

add_executable(sbwt_lcs_gpu
  # src/CLI/sbwt.cpp
  # src/CLI/sbwt_build.cpp
  # src/CLI/sbwt_search.cpp
  # src/CLI/sbwt_build_from_plain_matrix.cpp
  src/main.cpp
  src/utils.cpp
  src/kernels.cu
  src/gpu_utils.cu
  src/sanity_test_cpu.cpp
  src/sanity_test_gpu.cu
  ${SBWT_SOURCES})
set_target_properties(sbwt_lcs_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES 80)

target_include_directories(sbwt_lcs_gpu PRIVATE
  ${PROJECT_SOURCE_DIR}/include/
  # ${PROJECT_SOURCE_DIR}/include/gpu/
  ${PROJECT_SOURCE_DIR}/sdsl-lite/include
  # ${PROJECT_BINARY_DIR}/external/sdsl-lite/build/external/libdivsufsort/include
  # ${PROJECT_BINARY_DIR}/include
  # ${PROJECT_SOURCE_DIR}
  # ${PROJECT_SOURCE_DIR}/KMC/include)
  )
target_link_libraries(sbwt_lcs_gpu PRIVATE
  sdsl
  # ${BZIP2}
  # ${ZLIB}
  Threads::Threads
  ${CXX_FILESYSTEM_LIBRARIES}
  OpenMP::OpenMP_CXX
  # kmc_tools
  # kmc_core)
  )

## Build tests if doing a Debug build
# if (BUILD_TESTS)
#   add_subdirectory(${PROJECT_SOURCE_DIR}/googletest
#     ${PROJECT_BINARY_DIR}/external/googletest/build)
#   find_package(Threads REQUIRED)
#   set(THREADS_PREFER_PTHREAD_FLAG ON)
#   find_package(OpenMP REQUIRED)
#   find_library(GTEST NAMES gtest PATHS ${PROJECT_SOURCE_DIR}/external/googletest/build/lib/)

#   add_executable(sbwt_tests
#     tests/test_main.cpp
#     src/CLI/sbwt_build.cpp
#     src/CLI/sbwt_search.cpp
#     ${SBWT_SOURCES})
#   target_include_directories(sbwt_tests PRIVATE
#     ${PROJECT_SOURCE_DIR}/include/sbwt
#     ${PROJECT_SOURCE_DIR}/include/sbwt/libwheeler
#     ${PROJECT_SOURCE_DIR}/sdsl-lite/include
#     ${PROJECT_BINARY_DIR}/external/sdsl-lite/build/external/libdivsufsort/include
#     ${PROJECT_BINARY_DIR}/include
#     ${PROJECT_SOURCE_DIR}
#     ${PROJECT_SOURCE_DIR}/googletest/googletest/include
#     ${PROJECT_SOURCE_DIR}/KMC/include)
#   target_link_libraries(sbwt_tests PRIVATE
#     gtest
#     ${BZIP2}
#     ${ZLIB}
#     sdsl
#     Threads::Threads
#     OpenMP::OpenMP_CXX
#     ${CXX_FILESYSTEM_LIBRARIES}
#     kmc_tools
#     kmc_core)
# else()
# #  set_target_properties(gtest PROPERTIES EXCLUDE_FROM_ALL 1)
# #  set_target_properties(gtest_main PROPERTIES EXCLUDE_FROM_ALL 1)
# endif()

# add_library(sbwt_static STATIC ${SBWT_SOURCES})
# target_include_directories(sbwt_static PRIVATE
#   ${PROJECT_SOURCE_DIR}/include/sbwt/
#   ${PROJECT_SOURCE_DIR}/include/sbwt/libwheeler
#   ${PROJECT_SOURCE_DIR}/sdsl-lite/include
#   ${PROJECT_BINARY_DIR}/external/sdsl-lite/build/external/libdivsufsort/include
#   ${PROJECT_BINARY_DIR}/include
#   ${PROJECT_SOURCE_DIR}
#   ${PROJECT_SOURCE_DIR}/KMC/include)
# target_link_libraries(sbwt_static PRIVATE
#   sdsl
#   ${BZIP2}
#   ${ZLIB}
#   Threads::Threads
#   ${CXX_FILESYSTEM_LIBRARIES}
#   kmc_tools
#   kmc_core)

