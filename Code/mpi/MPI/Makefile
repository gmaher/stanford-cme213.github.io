CXX=mpic++
CXXFLAGS=

SRC := mpi_hello.cpp
EXE := mpi_hello

.PHONY: all
all: $(EXE)

# pattern rules
% : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	$(RM) $(EXE)
