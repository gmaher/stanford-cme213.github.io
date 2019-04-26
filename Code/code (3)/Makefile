CXX=nvcc
CXXFLAGS=-O3 --compiler-options -Wall
GPU=-arch=sm_35

SRC := $(wildcard *.cu)
EXE := $(patsubst %.cu,%,$(filter %.cu,$(SRC)))

.PHONY: all
all: $(EXE)

# pattern rules
% : %.cu
	$(CXX) $(GPU) $(CXXFLAGS) $< -o $@

memcheck_demo: memcheck_demo.cu
	$(CXX) $(GPU) -Xcompiler -rdynamic -lineinfo $< -o $@

.PHONY: clean
clean:
	$(RM) $(EXE)
