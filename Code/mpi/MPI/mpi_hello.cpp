#include "mpi.h"
#include <cstdio>
#include <cstdlib>

#define MASTER 0

int main(int argc, char *argv[])
{

  // Some MPI magic to get started
  MPI_Init(&argc, &argv);

  // How many processes are running
  int numtasks;
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  // What's my rank?
  int taskid;
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  // Which node am I running on?
  int len;
  char hostname[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(hostname, &len);

  printf("Hello from task %2d running on node: %s\n", taskid, hostname);

  // Only one processor will do this
  if (taskid == MASTER)
  {
    printf("MASTER process: the number of MPI tasks is: %2d\n", numtasks);
  }

  // Close down all MPI magic
  MPI_Finalize();

  return 0;
}
