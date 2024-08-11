# This file ensures that CMake does not run in a folder where there is an
# existing CMakeLists.txt, as this is usually not desired. It is better to build
# in a separate directory as CMake generates a lot of 'garbage' files for
# caching

file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
  message("######################################################")
  message("Warning: in-source builds are disabled")
  message("You cannot build in a source directory (or any directory with a CMakeLists.txt file)")
  message("Please create a separate build directory and run cmake from there")
  message("######################################################")
  message(FATAL_ERROR "Quitting configuration")
endif()
