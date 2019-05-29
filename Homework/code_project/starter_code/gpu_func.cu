#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

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

__global__
void gemm_gpu(float* A, float* B, float* C, float* D, int wA, int wB, float* alpha, float* beta){
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int a_start = wA*BLOCK_SIZE*by;
  int a_end   = a_start+wA-1;
  int a_step  = BLOCK_SIZE;

  int b_start = BLOCK_SIZE*bx;
  int b_step  = BLOCK_SIZE*wB;

  float Dsub = 0;

  for (int a = a_start, b=b_start; a <= a_end; a+=a_step, b+=b_step){

    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    Asub[ty][tx] = A[a + tx + wA*ty];
    Bsub[ty][tx] = B[b + tx + wB*ty];

    __syncthreads();

    for (int i = 0; i < BLOCK_SIZE; i++){
      Dsub += Asub[ty][i]*Bsub[ty][i];
    }
    __syncthreads();
  }

  int c = wB*BLOCK_SIZE*by+BLOCK_SIZE*bx;
  D[c + tx + ty*wB] = (*alpha)*Csub + (*beta)*C[c + tx + ty*wB];

}

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double* alpha, double* beta,
           int M, int N, int K) {
    /* TODO: Write an efficient GEMM implementation on GPU */

    return 1;
}
