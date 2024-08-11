# This enables or disables several CMake options, such as having Interprodecural
# optimisation, and positioning independent code

option(
  BUILD_SHARED_LIBS
  "Build shared libraries. Off by default (ie static is the default)"
  OFF
)

# suitable for shared libraries to make them executable from anywhere in memory
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# IPO = Interprocedural Optimization
# This is suitable when we have a lot of small functions which are frequently used
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
  option(
    IPO
    "Activate interprocedural optimization. Off by default"
    OFF
  )
endif()

function(activate_ipo THIS_TARGET)
  if(ipo_supported)
    set_target_properties(THIS_TARGET PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()
endfunction()

option(
  ENABLE_MARCH_NATIVE
  "Enable native compilation optimisations. Warning: removes compatibility with other architectures"
  OFF
)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE AND ENABLE_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
