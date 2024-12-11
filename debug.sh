#!/bin/bash

mkdir -p build
cd build || { echo "cd failed" && exit 1; }
#export compile commands to generate compile_commands.json
#march native for local optimization
cmake \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_PROFILING=OFF \
  -DENABLE_MARCH_NATIVE=OFF \
  ..
if [ $? -ne 0 ]; then >&2 echo "Cmake generation failed" && cd .. && exit 1; fi
cmake --build . -j8
if [ $? -ne 0 ]; then >&2 echo "Build failed" && cd .. && exit 1; fi
cd ..

#./build/bin/sbwt_lcs_gpu