#include <stdio.h>
#include <unistd.h>
#include <cstdlib>
#include "mpi.h"

#define MAX_LEN 1 << 18 /* maximum vector length	*/
#define TRIALS 100		/* trials for each msg length	*/
#define PROC_0 0		/* processor 0			*/
#define B0_TYPE 176		/* message "types"		*/
#define B1_TYPE 177

int main(int argc, char *argv[])
{
	int numprocs, p, /* number of processors, proc index	*/
		myid,		 /* this processor's "rank"		*/
		length,		 /* vector length			*/
		i, t;

	double b0[MAX_LEN], b1[MAX_LEN]; /* vectors			*/

	double start_time, end_time; /* "wallclock" times		*/

	MPI_Status stat; /* MPI structure containing return	*/
	/* codes for message passing operations */

	MPI_Request send_handle, recv_handle; /* For nonblocking msgs	*/

	MPI_Init(&argc, &argv); /* initialize MPI	*/

	int option = 0;
	char *str_end;

	int proc_1;

	char *cvalue = NULL;
	int c;

	opterr = 0;

	while ((c = getopt(argc, argv, "p:")) != -1)
	{
		switch (c)
		{
		case 'p':
			proc_1 = atoi(optarg);
			break;
		}
	}

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs); /*how many processors?	*/
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);	 /*which one am I?	*/

	if (myid == PROC_0)
		printf("proc_1 = %2d\n", proc_1);

	if (myid == PROC_0)
	{
		/* generate processor 0's vector */
		for (i = 0; i < MAX_LEN; ++i)
			b0[i] = i % 8;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	/* warmup, if necessary */
	for (length = 1; length <= MAX_LEN; length *= 2)
		if (myid == PROC_0)
		{
			MPI_Send(b0, length, MPI_DOUBLE, proc_1, B1_TYPE, MPI_COMM_WORLD);
			MPI_Recv(b1, length, MPI_DOUBLE, proc_1, B0_TYPE, MPI_COMM_WORLD, &stat);
		}
		else if (myid == proc_1)
		{
			MPI_Recv(b1, length, MPI_DOUBLE, PROC_0, B1_TYPE, MPI_COMM_WORLD, &stat);
			MPI_Send(b0, length, MPI_DOUBLE, PROC_0, B0_TYPE, MPI_COMM_WORLD);
		}

	/* measure message passing speed for vectors of various lengths */

	for (length = MAX_LEN; length <= MAX_LEN; length *= 2)
	{
		MPI_Barrier(MPI_COMM_WORLD);

		if (myid == PROC_0)
			start_time = MPI_Wtime();

		for (t = 0; t < TRIALS; ++t)
		{
			if (myid == PROC_0)
			{
				MPI_Send(b0, length, MPI_DOUBLE, proc_1, B1_TYPE, MPI_COMM_WORLD);
				MPI_Recv(b1, length, MPI_DOUBLE, proc_1, B0_TYPE, MPI_COMM_WORLD, &stat);
			}
			else if (myid == proc_1)
			{
				MPI_Recv(b1, length, MPI_DOUBLE, PROC_0, B1_TYPE, MPI_COMM_WORLD, &stat);
				MPI_Send(b0, length, MPI_DOUBLE, PROC_0, B0_TYPE, MPI_COMM_WORLD);
			}
		}
		if (myid == PROC_0)
		{
			end_time = MPI_Wtime();
			printf("Length = %8d\tAverage time [mu-sec] = %10.8f\n",
				   length, 1e6 * (end_time - start_time) / (double)(2 * TRIALS * length * 8));
		}
	}

	MPI_Finalize();
}
