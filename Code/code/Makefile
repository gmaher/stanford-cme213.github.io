LIB := -lpthread
CC := gcc
CPP := g++
CFLAGS := -Wall
CPPFLAGS := -std=c++11 -Wall

SRC := $(wildcard *.c) $(wildcard *.cpp)
#SRC := cpp_thread.cpp
EXE := $(patsubst %.c,%,$(filter %.c,$(SRC))) $(patsubst %.cpp,%,$(filter %.cpp,$(SRC)))

.PHONY: all
all: $(EXE)

# pattern rules
% : %.c
	$(CC) $(CFLAGS) -o $@ $< $(LIB)

% : %.cpp
	$(CPP) $(CPPFLAGS) -o $@ $< $(LIB)

.PHONY: clean
clean:
	$(RM) $(EXE)
