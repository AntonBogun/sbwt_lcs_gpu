REL_INCLUDE_PATH = ../include
INCLUDE_PATH = include
SOURCE_PATH = src/search.cpp
BUILD_PATH = build/search
CREATE_PATH = mkdir -p build

NVCC_FLAGS = -Xcompiler -fopenmp -lgomp -gencode arch=compute_80,code=sm_80 -O3 --include-path $(REL_INCLUDE_PATH) -x cu -g
# NVCC_FLAGS = -Xcompiler -fopenmp -lgomp -Xcompiler -Wall -Xcompiler -Wextra -gencode arch=compute_80,code=sm_80 -O3 --include-path $(REL_INCLUDE_PATH) -x cu -g

NVCC = nvcc

.ONESHELL:

all: $(BUILD_PATH)

$(BUILD_PATH): $(SOURCE_PATH) $(wildcard $(INCLUDE_PATH)/**/*.hpp)
	module load gcc
	module load cuda
	$(CREATE_PATH)
	$(MAKE) clean
	$(NVCC) $(NVCC_FLAGS) -o $@ $(SOURCE_PATH)
	cuobjdump -ptx $@ > $@.ptx
	cuobjdump -sass $@ > $@.sass

clean:
	rm -f $(BUILD_PATH) $(BUILD_PATH).ptx $(BUILD_PATH).sass

