CXX=nvcc
CXXFLAGS=-O3 --compiler-options -Wall -arch=sm_35

SRC := $(wildcard *.cu)
EXE := $(patsubst %.cu,%,$(filter %.cu,$(SRC)))

.PHONY: all
all: $(EXE)

# pattern rules
% : %.cu
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	$(RM) $(EXE)
