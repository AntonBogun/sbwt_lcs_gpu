# Adds the flags -pg to gcc so we can profile with gprof

option(
  ENABLE_PROFILING
  "Add -pg flags so that profiling is enables for gprof to use"
  OFF
)

if (ENABLE_PROFILING)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif()
