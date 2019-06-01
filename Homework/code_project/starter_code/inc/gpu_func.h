#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

class NeuralNetworkGPU {
public:
  int n_feats;
  int n_classes;
  int n_batch;
  int n_hidden;

  double* Xd;
  double* Yd;

  double* W1_d;
  double* b1_d;
  double* z1_d;
  double* a1_d;

  double* W2_d;
  double* b2_d;
  double* z2_d;
  double* a2_d;

  double* yhat_d;

  NeuralNetworkGPU(int x_size, int y_size, int hidden_size, int batch_size) {
      n_feats   = x_size;
      n_classes = y_size;
      n_batch   = batch_size;
      n_hidden  = hidden_size;

      cudaMalloc((void**)&Xd, sizeof(double)*x_size*batch_size);
      cudaMalloc((void**)&Yd, sizeof(double)*y_size*y_size);

      int h1 = nn.W[0].n_rows;
      int w1 = nn.W[0].n_cols;
      cudaMalloc((void**)&W1_d, sizeof(double)*h1*w1);
      cudaMalloc((void**)&b1_d, sizeof(double)*h1);
      cudaMalloc((void**)&z1_d, sizeof(double)*h1*batch_size);
      cudaMalloc((void**)&a1_d, sizeof(double)*h1*batch_size);
      std::cout << "layer 1 h=" << h1 << ", w=" << w1 << "\n";

      int h2 = nn.W[1].n_rows;
      int w2 = nn.W[1].n_cols;
      cudaMalloc((void**)&W2_d, sizeof(double)*h2*w2);
      cudaMalloc((void**)&b2_d, sizeof(double)*h2);
      cudaMalloc((void**)&z2_d, sizeof(double)*h2*batch_size);
      cudaMalloc((void**)&a2_d, sizeof(double)*h2*batch_size);
      std::cout << "layer 2 h=" << h2 << ", w=" << w2 << "\n";
    }

};

struct event_pair {
    cudaEvent_t start;
    cudaEvent_t end;
};

inline void check_launch(const char* kernel_name) {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        std::cerr << "error in " << kernel_name << " kernel" << std::endl;
        std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

inline void start_timer(event_pair* p) {
    cudaEventCreate(&p->start);
    cudaEventCreate(&p->end);
    cudaEventRecord(p->start, 0);
}


inline double stop_timer(event_pair* p) {
    cudaEventRecord(p->end, 0);
    cudaEventSynchronize(p->end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, p->start, p->end);
    cudaEventDestroy(p->start);
    cudaEventDestroy(p->end);
    return elapsed_time;
}

int useless_gpu_add_one(int t);

int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K);

int mySigmoid(double* S, double* X, int M, int N);
int myHadamard(double* X, double* Y, double* H, int M, int N);
int myTranspose(double* X, double* Xt, int M, int N);
int myMatAdd(double* X, double* Y, double* Z, int M, int N, double alpha);
int mySoftmax(double* X, double* S, int M, int N);

#endif
