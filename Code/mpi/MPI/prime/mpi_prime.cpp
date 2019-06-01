#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

/* The largest integer we will consider */
#define LIMIT 40000000

/* Rank of the master task */
#define MASTER 0

int IsPrime(int n)
{
  if (n >= 11)
  {
    int squareroot = (int)sqrt(float(n));

    for (int i = 3; i <= squareroot; i += 2)
      if ((n % i) == 0)
      {
        return 0;
      }

    return 1;
  }
  else if (n == 2 || n == 3 || n == 5 || n == 7)
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

int main(int argc, char *argv[])
{
  int ntasks;                            /* total number of tasks in partition */
  int rank;                              /* task identifier */
  int pcsum;                             /* number of primes found by all tasks */
  int maxprime;                          /* largest prime found */
  int len;                               /* length of hostname */
  char hostname[MPI_MAX_PROCESSOR_NAME]; /* hostname */

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(hostname, &len);
  printf("MPI task %d has started on %s [total number of processors %d]\n",
         rank, hostname, ntasks);

  /* task with rank MASTER does this part */
  if (rank == MASTER)
  {
    printf("Using %d tasks to scan %d numbers...\n", ntasks, LIMIT);
  }

  double start_time = MPI_Wtime(); /* Initialize start time */

  int mystart = (rank * 2) + 3; /* Starting point - must be odd number */
  int stride = ntasks * 2;      /* Determine stride, skipping even numbers */
  int pc = 0;                   /* Prime counter */
  int foundone = 0;             /* Last prime that was found */

  if (rank == MASTER)
  {
    pc = 1;
    foundone = 2; /* Don't forget about 2; it is a prime! */
    //    printf("Task %d: %d\n", rank, foundone);
  }

  for (int n = mystart; n <= LIMIT; n += stride)
  {
    if (IsPrime(n))
    {
      pc++;         // found a prime
      foundone = n; // last prime that we have found
      /* Optional: print each prime as it is found */
      //      printf("Task %d: %d\n", rank, foundone);
    }
  }

  // Total number of primes found by all processes:     MPI_SUM
  MPI_Reduce(&pc, &pcsum, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);

  // The largest prime that was found by all processes: MPI_MAX
  MPI_Reduce(&foundone, &maxprime, 1, MPI_INT, MPI_MAX, MASTER, MPI_COMM_WORLD);

  double end_time = MPI_Wtime();

  if (rank == MASTER)
  {
    printf("Done.\nLargest prime is %d.\nTotal number of primes found: %d\n",
           maxprime, pcsum);
    printf("Wall clock time elapsed: %.2lf seconds\n", end_time - start_time);
  }

  MPI_Finalize();

  return 0;
}

#if 0

for(int n=mystart; n<=LIMIT; n+=stride) {
  if(IsPrime(n)) {
    pc++;         // found a prime
    foundone = n; // last prime that we have found
  }
}
// Total number of primes found by all processes:     MPI_SUM
MPI_Reduce(&pc, &pcsum, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
// The largest prime that was found by all processes: MPI_MAX
MPI_Reduce(&foundone, &maxprime, 1, MPI_INT, MPI_MAX, MASTER, MPI_COMM_WORLD);

#endif
