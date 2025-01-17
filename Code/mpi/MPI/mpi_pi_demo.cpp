#include "mpi.h"
#include <cstdio>
#include <cstdlib>

#define DARTS 500000 /* number of throws at dartboard */
#define ROUNDS 10    /* number of times "darts" is iterated */
#define MASTER 0     /* task ID of master task */

#define sqr(x) ((x) * (x))

/*
  Explanation of constants and variables used in this function:
  darts       = number of throws at dartboard
  score       = number of darts that hit circle
  n           = index variable
  r           = random number scaled between 0 and 1
  x_coord     = x coordinate, between -1 and 1
  x_sqr       = square of x coordinate
  y_coord     = y coordinate, between -1 and 1
  y_sqr       = square of y coordinate
  pi          = computed value of pi
 */
double DartBoard(int darts)
{
  int score = 0;

  /* "throw darts at board" */
  for (int n = 1; n <= darts; n++)
  {
    /* generate random numbers for x and y coordinates */
    double r = (double)rand() / (double)(RAND_MAX);
    double x_coord = (2.0 * r) - 1.0;
    r = (double)rand() / (double)(RAND_MAX);
    double y_coord = (2.0 * r) - 1.0;

    /* if dart lands in circle, increment score */
    if ((sqr(x_coord) + sqr(y_coord)) <= 1.0)
    {
      score++;
    }
  }

  /* estimate pi */
  return 4.0 * (double)score / (double)darts;
}

int main(int argc, char *argv[])
{
  /* Obtain number of tasks and task ID */
  MPI_Init(&argc, &argv);
  int taskid,   /* task ID - also used as seed number */
      numtasks, /* number of tasks */
      len;      /* length of hostname (no. of chars) */
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  char hostname[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(hostname, &len);
  printf("MPI task %2d has started on %s [total number of processors %d]\n",
         taskid, hostname, numtasks);

  /* Set seed for random number generator equal to task ID */
  srandom(2017 + (taskid << 4));

  double avepi = 0.;

  for (int i = 0; i < ROUNDS; i++)
  {
    /* All tasks calculate pi using the dartboard algorithm */
    double my_pi = DartBoard(DARTS);

    /* Workers send my_pi to master */
    /* Message tag is set to the iteration count */
    if (taskid != MASTER)
    {
      int tag = i;
      /* TODO: send my_pi to process MASTER
      Use tag to make sure the message corresponds to the right round
      The last argument is MPI_COMM_WORLD.
      Check MPI_Recv(...) below for a hint.
      Compile and run the code using 8 processes:
      mpirun -mca btl ^openib -n 8 ./mpi_pi_demo
      */
      // int rc = MPI_Send(...);

      if (rc != MPI_SUCCESS)
      {
        printf("%d: Send failure on round %d\n", taskid, tag);
      }
    }
    else
    {

      /* Master receives messages from all workers */
      /* Message tag is equal to the iteration count */
      /* Message source is set to the wildcard MPI_ANY_SOURCE: */
      /*  a message can be received from any task, as long as the */
      /*  message types match */
      int tag = i;
      double pisum = 0;

      for (int n = 1; n < numtasks; n++)
      {
        double pirecv;
        MPI_Status status;
        int rc = MPI_Recv(&pirecv, 1, MPI_DOUBLE, MPI_ANY_SOURCE,
                          tag, MPI_COMM_WORLD, &status);

        if (rc != MPI_SUCCESS)
        {
          printf("%d: Receive failure on round %d\n", taskid, tag);
        }

        /* Running total of pi */
        pisum += pirecv;
      }

      /* Master calculates the average value of pi for this iteration */
      double pi = (pisum + my_pi) / numtasks;
      //printf("   pi for this round = %10.8f\n",
      //       pi);
      /* Master calculates the average value of pi over all iterations */
      avepi = ((avepi * i) + pi) / (i + 1);
      printf("   After %8d throws, average value of pi = %10.8f\n",
             (DARTS * (i + 1) * numtasks), avepi);
    }
  }

  if (taskid == MASTER)
  {
    printf("\nExact value of pi: 3.1415926535897 \n");
  }

  MPI_Finalize();

  return 0;
}

#if 0
for(int i = 0; i < ROUNDS; i++) {
  double my_pi = DartBoard(DARTS);
  if(taskid != MASTER) {
    int tag = i;
    int rc = MPI_Send(&my_pi, 1, MPI_DOUBLE,
                      MASTER, tag, MPI_COMM_WORLD);
  } else {
    int tag = i;
    double pisum = 0;
    for(int n = 1; n < numtasks; n++) {
      double pirecv;
      MPI_Status status;
      int rc = MPI_Recv(&pirecv, 1, MPI_DOUBLE, MPI_ANY_SOURCE,
                        tag, MPI_COMM_WORLD, &status);
      pisum += pirecv;
    }
    double pi = (pisum + my_pi)/numtasks;
    printf("   pi for this round = %10.8f\n", pi);
    avepi = ((avepi * i) + pi)/(i + 1);
    printf("   After %8d throws, average value of pi = %10.8f\n",
           (DARTS * (i + 1) * numtasks), avepi);
  }
}
#endif
