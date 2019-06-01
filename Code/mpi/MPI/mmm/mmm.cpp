#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <vector>
using std::vector;
using std::abs;

// Matrix A
float AEntry(int i, int j, int n) {
  if(i%2 == 0) {
    return 1;
  } else if(j%3 == 0) {
    return -1;
  } else if(j%3 == 1) {
    return 2;
  } else {
    return -2;
  }
}

// Matrix B
float BEntry(int i, int j, int n) {
  return i * n - j;
}

void MatrixMultiply(const int n, vector<double>& a, vector<double>& b,
                    vector<double>* c) {
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      double cs = 0.;

      for(int k = 0; k < n; k++) {
        cs += a.at(i*n + k) * b.at(k*n + j);
      }

      c->at(i*n + j) += cs;
    }
  }
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc,&argv);

  /* Get the communicator related information */
  int nprocs, myrank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  const unsigned n = 512*3;

  /* Set up the Cartesian topology */
  int dims[2], periods[2];
  dims[0] = dims[1] = int(sqrt(nprocs));
  assert(dims[0] * dims[1] == nprocs);
  assert(n % dims[0] == 0);

  /* Set the periods for wraparound connections */
  periods[0] = periods[1] = 1;

  /* Create the Cartesian topology, with rank reordering */
  MPI_Comm comm_2d;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_2d);

  /* Get the rank and coordinates with respect to the new topology */
  int my2drank, mycoords[2];
  MPI_Comm_rank(comm_2d, &my2drank);
  MPI_Cart_coords(comm_2d, my2drank, 2, mycoords);

  /* Compute ranks of the up and left shifts */
  int uprank, downrank, leftrank, rightrank;
  MPI_Cart_shift(comm_2d, 1, 1, &leftrank, &rightrank); // source, destination
  MPI_Cart_shift(comm_2d, 0, 1, &uprank,   &downrank);  // source, destination

  /* Determine the dimension of the local matrix block */
  const unsigned nlocal = n/dims[0];

  if(myrank == 0) {
    printf("Dimension of matrix: %d, block size: %d, number of procs along both dims: %d %d\n",
           n, nlocal, dims[0], dims[1]);
  }

  assert(nlocal > 0 && nlocal <= n);
  assert(nlocal*dims[0] == n);

  /* Setup the a_buffers and b_buffers arrays.
   * We need two buffers for communication. */
  vector<double> a_buffers[2], b_buffers[2];
  vector<double> c(nlocal*nlocal);

  const int i_offset = mycoords[0] * nlocal;
  const int j_offset = mycoords[1] * nlocal;

  a_buffers[0].resize(nlocal*nlocal);
  b_buffers[0].resize(nlocal*nlocal);

  for(unsigned i=0; i < nlocal; ++i)
    for(unsigned j=0; j < nlocal; ++j) {
      a_buffers[0].at(i * nlocal + j) = AEntry(i+i_offset, j+j_offset, n);
      b_buffers[0].at(i * nlocal + j) = BEntry(i+i_offset, j+j_offset, n);
      c.at(i * nlocal + j) = 0.;
    }

  a_buffers[1].resize(nlocal*nlocal);
  b_buffers[1].resize(nlocal*nlocal);

  // For timing.
  double starttime, endtime;
  starttime = MPI_Wtime();

  /* Perform the initial matrix alignment. First for A and then for B */
  {
    int shiftsource, shiftdest;
    MPI_Status status;
    MPI_Cart_shift(comm_2d, 1, -mycoords[0], &shiftsource, &shiftdest);
    // Shift A
    MPI_Sendrecv_replace(&(a_buffers[0][0]), nlocal*nlocal, MPI_DOUBLE, shiftdest,
                         1, shiftsource, 1, comm_2d, &status);
    /* Sends and receives using a single buffer */

    MPI_Cart_shift(comm_2d, 0, -mycoords[1], &shiftsource, &shiftdest);
    // Shift B
    MPI_Sendrecv_replace(&(b_buffers[0][0]), nlocal*nlocal, MPI_DOUBLE, shiftdest,
                         1, shiftsource, 1, comm_2d, &status);
  }

  MPI_Request reqs[4];

  /* Get into the main computation loop */
  int idim;

  for(idim=0; idim < dims[0]; idim++) {
    // We use non-blocking communications
    MPI_Isend(&(a_buffers[idim%2][0]),     nlocal*nlocal, MPI_DOUBLE, leftrank,  1,
              comm_2d, &reqs[0]);
    MPI_Isend(&(b_buffers[idim%2][0]),     nlocal*nlocal, MPI_DOUBLE, uprank,    1,
              comm_2d, &reqs[1]);
    MPI_Irecv(&(a_buffers[(idim+1)%2][0]), nlocal*nlocal, MPI_DOUBLE, rightrank, 1,
              comm_2d, &reqs[2]);
    MPI_Irecv(&(b_buffers[(idim+1)%2][0]), nlocal*nlocal, MPI_DOUBLE, downrank,  1,
              comm_2d, &reqs[3]);
    /* idim%2 and (idim+1)%2 allow us to ping-pong between the two buffers. */

    /* C = C + A*B */
    MatrixMultiply(nlocal, a_buffers[idim%2], b_buffers[idim%2], &c);
    // Communications are happening while we are computing

    // We need to wait for communications to end before moving on
    // to the next iteration.
    for(unsigned j=0; j<4; j++) {
      MPI_Status status;
      MPI_Wait(&reqs[j], &status);
    }
  }

  // At the end of the loop, idim == dims[0]. This is needed below.

  endtime = MPI_Wtime();

  if(myrank == 0) {
    const double etime = endtime-starttime;
    printf("The calculation took %f seconds\n",etime);
    printf("p x runtime = %f \n",nprocs * etime);
  }

  // Check result: C == A*B
  {
    vector<double> d(nlocal*nlocal);

    for(unsigned i = 0; i < nlocal; i++) {
      for(unsigned j = 0; j < nlocal; j++) {
        double cs = 0.;

        for(unsigned k = 0; k < n; k++) {
          cs += AEntry(i+i_offset, k, n) * BEntry(k, j+j_offset, n);
        }

        assert(cs == c.at(i*nlocal + j));
        // Result is exact because of integer arithmetic.
      }
    }
  }

  /* Restore the original distribution of A and B (an optional step) */
  {
    int shiftsource, shiftdest;
    MPI_Status status;
    MPI_Cart_shift(comm_2d, 1, mycoords[0], &shiftsource, &shiftdest);
    // Shift A
    MPI_Sendrecv_replace(&(a_buffers[idim%2][0]), nlocal*nlocal, MPI_DOUBLE,
                         shiftdest, 1, shiftsource, 1, comm_2d, &status);

    MPI_Cart_shift(comm_2d, 0, mycoords[1], &shiftsource, &shiftdest);
    // Shift B
    MPI_Sendrecv_replace(&(b_buffers[idim%2][0]), nlocal*nlocal, MPI_DOUBLE,
                         shiftdest, 1, shiftsource, 1, comm_2d, &status);
  }

  // Check that A and B are back to normal
  {
    for(unsigned i = 0; i < nlocal; i++) {
      for(unsigned j = 0; j < nlocal; j++) {
        assert(a_buffers[idim%2][i*nlocal + j] == AEntry(i+i_offset, j+j_offset, n));
        assert(b_buffers[idim%2][i*nlocal + j] == BEntry(i+i_offset, j+j_offset, n));
      }
    }
  }

  // We passed all the tests! We are done.

  MPI_Comm_free(&comm_2d); /* Free up communicator */

  MPI_Barrier(MPI_COMM_WORLD);
  // Making sure that all tests have passed before printing message

  if(myrank == 0) {
    printf("All tests were successfully passed.\n");
  }

  MPI_Finalize();

  return 0;
}
