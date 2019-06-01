EXECS=ring_DL ring_NS ring_SEC ring_SR
MPICC=mpicc
FLAGS = -Wall

all: ${EXECS}

ring_DL: ring_DL.c
	${MPICC} $(FLAGS) -o $@ $<

ring_NS: ring_NS.c
	${MPICC} $(FLAGS) -o $@ $<

ring_SEC: ring_SEC.c
	${MPICC} $(FLAGS) -o $@ $<

ring_SR: ring_SR.c
	${MPICC} $(FLAGS) -o $@ $<

clean:
	rm -f ${EXECS}
