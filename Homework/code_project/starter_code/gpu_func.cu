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
      Dsub += Asub[ty][i]*Bsub[i][tx];
    }
    __syncthreads();
  }

  int c = wB*BLOCK_SIZE*by+BLOCK_SIZE*bx;
  D[c + tx + ty*wB] = (*alpha)*Dsub + (*beta)*C[c + tx + ty*wB];

}

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double* alpha, double* beta,
           int M, int N, int K) {
    /* TODO: Write an efficient GEMM implementation on GPU */

    float* Ad;
    int a_size = M*N*sizeof(float);
    cudaMalloc((void**)&Ad, a_size);
    cudaMemcpy(Ad, A, a_size, cudaMemcpyHostToDevice);

    float* Bd;
    int b_size = N*K*sizeof(float);
    cudaMalloc((void**)&Bd, b_size);
    cudaMemcpy(Bd, B, b_size, cudaMemcpyHostToDevice);

    float* Cd;
    int C_size = M*K*sizeof(float);
    cudaMalloc((void**)&Cd, c_size);
    cudaMemcpy(Cd, C, c_size, cudaMemcpyHostToDevice);

    float* Dd;
    cudaMalloc((void**)&Dd, c_size);

    dim3 dimBlock(32,32);
    dim3 dimGrid(K/dimBlock.x, M/dimBlock.y);

    gemm_gpu<<<dimGrid, dimBlock>>(Ad, Bd, Cd, Dd, N, K, alpha, beta);

    cudaMemcpy(D, Dd, c_size, cudaMemcpyDeviceToHost);

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    cudaFree(Dd);  
    return 1;
}
