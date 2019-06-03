#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

#define BLOCK_SIZE 32

__global__
void device_add_one(int* d_result, int t) {
    *d_result = t + 1;
}

/*
Just a dummy function that can be used to warm up GPU
*/
int useless_gpu_add_one(int t) {
    int result;
    int* d_result;

    checkCudaErrors(cudaMalloc((void**)&d_result, 1 * sizeof(int)));

    event_pair timer;
    start_timer(&timer);
    device_add_one<<<1,1>>>(d_result, t);
    check_launch("device_add_one");
    double time = stop_timer(&timer);

    std::cout << "device_add_one took: " << time << " seconds" << std::endl;

    checkCudaErrors(cudaMemcpy(&result, d_result, 1 * sizeof(int),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_result));
    return result;
}

// __global__
// void gemm_gpu(double* A, double* B, double* C, double* D, int hA, int wA,
//     int hB, int wB, double alpha, double beta){
//   int bx = blockIdx.x;
//   int by = blockIdx.y;
//
//   int tx = threadIdx.x;
//   int ty = threadIdx.y;
//
//   int a_start = wA*BLOCK_SIZE*by;
//   int a_end   = a_start+wA-1;
//   int a_step  = BLOCK_SIZE;
//
//   int b_start = BLOCK_SIZE*bx*wB;
//   int b_step  = BLOCK_SIZE*wB;
//
//   float Dsub = 0;
//
//   for (int a = a_start, b=b_start; a <= a_end; a+=a_step, b+=b_step){
//
//     __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
//     __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];
//
//     if (bx*BLOCK_SIZE+tx >= wA || bx*BLOCK_SIZE+ty >= hB ||
//       by*BLOCK_SIZE+ty >= hA || by*BLOCK_SIZE+tx >= wB){
//       Asub[ty][tx] = 0;
//       Bsub[ty][tx] = 0;
//     }
//     else{
//       //printf("a: %u %u %u %u\n", a,tx,ty,a + tx + wA*ty);
//       //printf("b: %u %u %u %u\n", b,tx,ty,b + tx + wA*ty);
//       Asub[ty][tx] = A[a + tx + wA*ty];
//       Bsub[ty][tx] = B[b + tx + wB*ty];
//     }
//
//     __syncthreads();
//
//     for (int i = 0; i < BLOCK_SIZE; i++){
//       Dsub += Asub[ty][i]*Bsub[i][tx];
//     }
//     __syncthreads();
//   }
//
//   if (bx*BLOCK_SIZE+tx < wB && by*BLOCK_SIZE+ty < hA){
//     int c = wB*BLOCK_SIZE*by+BLOCK_SIZE*bx;
//     //printf("c: %u %u %u %u\n", c,tx,ty,c + tx + ty*wB);
//     D[c + tx + ty*wB] = alpha*Dsub + beta*C[c + tx + ty*wB];
//   }
//
// }

__global__
void gemm_gpu(double* A, double* B, double* C, double* D, int hA, int wA,
    int hB, int wB, double alpha, double beta){
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by*BLOCK_SIZE+ty;
  int col = bx*BLOCK_SIZE+tx;

  int a_start = row;
  int a_step  = hA;
  int b_start = col*hB;

  double entry = 0;

  if (row < hA && col < wB){
    for (int i = 0; i < hB; i++){
      entry += A[a_start+i*a_step]*B[b_start+i];
    }
    D[row+col*hA] = alpha*entry + beta*C[row+col*hA];
  }

}

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double* alpha, double* beta,
           int M, int N, int K) {
    /* TODO: Write an efficient GEMM implementation on GPU */


    int c_size = M*N*sizeof(double);

    double* Dd;
    cudaMalloc((void**)&Dd, c_size);

    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(N/dimBlock.x+1, M/dimBlock.y+1);

    gemm_gpu<<<dimGrid, dimBlock>>>(A, B, C, Dd, M, K, K, N,
      *alpha, *beta);

    cudaMemcpy(C, Dd, c_size, cudaMemcpyDeviceToDevice);

    cudaFree(Dd);
    return 0;
}

__global__
void sigmoid_gpu(double* __restrict__ X, double* __restrict__ S, int M, int N){
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by*BLOCK_SIZE+ty;
  int col = bx*BLOCK_SIZE+tx;

  int id = col*M+row;

  if (row < M && col < N){
    S[id] = 1.0/(1.0+exp(-X[id]));
    // if (S[id] >= 0){
    //   S[id] = 1.0/(exp(-X[id])+1);
    // }else{
    //   S[id] = exp(X[id])/(exp(X[id])+1);
    // }
  }

}

int mySigmoid(double* __restrict__ X, double* __restrict__ S, int M, int N) {
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(N/dimBlock.x+1, M/dimBlock.y+1);

    sigmoid_gpu<<<dimGrid, dimBlock>>>(X, S, M,N);
    return 0;
}

__global__
void hadamard_gpu(double* __restrict__ X, double* __restrict__ Y, double* __restrict__ H, int M, int N){
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by*BLOCK_SIZE+ty;
  int col = bx*BLOCK_SIZE+tx;

  int id = col*M+row;

  if (row < M && col < N){
    H[id] = X[id]*Y[id];
  }

}

int myHadamard(double* __restrict__ X, double* __restrict__ Y, double* __restrict__ H, int M, int N) {
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(N/dimBlock.x+1, M/dimBlock.y+1);

    hadamard_gpu<<<dimGrid, dimBlock>>>(X, Y, H, M, N);
    return 0;
}

__global__
void transpose_gpu(double* __restrict__ X, double* __restrict__ Xt, int M, int N){
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by*BLOCK_SIZE+ty;
  int col = bx*BLOCK_SIZE+tx;

  int id1 = col*M+row;
  int id2 = row*N+col;

  if (row < M && col < N){
    Xt[id2] = X[id1];
  }

}

int myTranspose(double* __restrict__ X, double* __restrict__ Xt, int M, int N) {
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(N/dimBlock.x+1, M/dimBlock.y+1);

    transpose_gpu<<<dimGrid, dimBlock>>>(X, Xt, M, N);
    return 0;
}

__global__
void matrix_add_gpu(double* __restrict__ X, double* __restrict__ Y, double* __restrict__ Z,
   int M, int N, double alpha){
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by*BLOCK_SIZE+ty;
  int col = bx*BLOCK_SIZE+tx;

  int id = col*M+row;

  if (row < M && col < N){
    Z[id] = X[id]+alpha*Y[id];
  }

}

int myMatAdd(double* __restrict__ X, double* __restrict__ Y, double* __restrict__ Z, int M, int N, double alpha) {
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(N/dimBlock.x+1, M/dimBlock.y+1);

    matrix_add_gpu<<<dimGrid, dimBlock>>>(X, Y, Z, M, N, alpha);
    return 0;
}

__global__
void row_sum_gpu(double* __restrict__ X, double* __restrict__ S,
  int M, int N, double alpha){

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by*BLOCK_SIZE+ty;
  int col = bx*BLOCK_SIZE+tx;

  int id = col*M+row;

  if (row < M && col < N){
    S[id] = 0;
    for(int i = row; i < M*N; i+=M){
      S[id] += X[i];
    }
    S[id] *= alpha;
  }

}

int myRowSum(double* __restrict__ X, double* __restrict__ S, int M, int N, double alpha) {
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(N/dimBlock.x+1, M/dimBlock.y+1);

    row_sum_gpu<<<dimGrid, dimBlock>>>(X, S, M, N, alpha);
    return 0;
}

__global__
void softmax_gpu(double* __restrict__ X, double* __restrict__ S, int M, int N){
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by*BLOCK_SIZE+ty;
  int col = bx*BLOCK_SIZE+tx;

  int id = col*M+row;

  if (row < M && col < N){
    double sum = 0;

    for (int i = col*M; i<(col+1)*M; i++){
      sum += exp(X[i]);
    }

    S[id] = exp(X[id])/sum;
  }

}

int mySoftmax(double* __restrict__ X, double* __restrict__ S, int M, int N) {
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(N/dimBlock.x+1, M/dimBlock.y+1);

    softmax_gpu<<<dimGrid, dimBlock>>>(X, S, M, N);
    return 0;
}

__global__
void print_gpu(double* __restrict__ X, int M, int N, int m, int n){
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by*BLOCK_SIZE+ty;
  int col = bx*BLOCK_SIZE+tx;

  int id = col*M+row;

  if (row < m && col < n){
    printf("%u,%u,%u: %f \n",row, col, id, X[id]);
  }

}

int myPrintMat(double* __restrict__ X, int M, int N, int m, int n) {
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(N/dimBlock.x+1, M/dimBlock.y+1);

    print_gpu<<<dimGrid, dimBlock>>>(X, M, N, m, n);
    return 0;
}
