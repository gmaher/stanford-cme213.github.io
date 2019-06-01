#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <armadillo>

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
int myPrintMat(double* X, int M, int N, int m, int n);

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
	printf("feats=%u, classes=%u, batch=%u, hidden=%u",n_feats, n_classes, n_batch, n_hidden);
       cudaMalloc((void**)&Xd, sizeof(double)*x_size*batch_size);
       cudaMalloc((void**)&Yd, sizeof(double)*y_size*y_size);

       int h1 = n_hidden;
       int w1 = n_feats;
       cudaMalloc((void**)&W1_d, sizeof(double)*h1*w1);
       cudaMalloc((void**)&b1_d, sizeof(double)*h1);
       cudaMalloc((void**)&z1_d, sizeof(double)*h1*batch_size);
       cudaMalloc((void**)&a1_d, sizeof(double)*h1*batch_size);
       std::cout << "layer 1 h=" << h1 << ", w=" << w1 << "\n";

       int h2 = n_classes;
       int w2 = n_hidden;
       cudaMalloc((void**)&W2_d, sizeof(double)*h2*w2);
       cudaMalloc((void**)&b2_d, sizeof(double)*h2);
       cudaMalloc((void**)&z2_d, sizeof(double)*h2*batch_size);
       cudaMalloc((void**)&a2_d, sizeof(double)*h2*batch_size);
       std::cout << "layer 2 h=" << h2 << ", w=" << w2 << "\n";
     }

   void set_weights(const arma::mat& W1, const arma::mat& b1, const arma::mat& W2, const arma::mat& b2){
	const double* w1_ptr = W1.memptr();
	const double* b1_ptr = b1.memptr();
	const double* w2_ptr = W2.memptr();
	const double* b2_ptr = b2.memptr();

	cudaMemcpy(W1_d, w1_ptr, sizeof(double)*n_hidden*n_feats, cudaMemcpyHostToDevice);
	cudaMemcpy(b1_d, b1_ptr, sizeof(double)*n_hidden, cudaMemcpyHostToDevice);
	cudaMemcpy(W2_d, w2_ptr, sizeof(double)*n_classes*n_hidden, cudaMemcpyHostToDevice);
	cudaMemcpy(b2_d, b2_ptr, sizeof(double)*n_classes, cudaMemcpyHostToDevice);

//	std::cout << "w1\n";
//	myPrintMat(W1_d, n_hidden, n_feats, 3, 3);
//      std::cout << "w1[0,0]=" << W1(0,0) << "\n";
//	std::cout << "w1[0,1]=" << W1(0,1) << "\n";
//	std::cout << "w1[0,2]=" << W1(0,2) << "\n";
//
//	std::cout << "w1[1,0]=" << W1(1,0) << "\n";
//	std::cout << "w1[1,1]=" << W1(1,1) << "\n";
//	std::cout << "w1[1,2]=" << W1(1,2) << "\n";
//
//	std::cout << "w1[2,0]=" << W1(2,0) << "\n";
//	std::cout << "w1[2,1]=" << W1(2,1) << "\n";
//	std::cout << "w1[2,2]=" << W1(2,2) << "\n";
//	std::cout << "b1\n";
//	myPrintMat(b1_d, n_hidden, 1, 3,1);
//      std::cout << "b1[0]=" << b1(0,0) << "\n";
//      std::cout << "b1[1]=" << b1(1,0) << "\n";
//      std::cout << "b1[2]=" << b1(2,0) << "\n";
//	std::cout << "w2\n";
//	myPrintMat(W2_d, n_classes, n_hidden, 3,3);

//      std::cout << "w2[0,0]=" << W2(0,0) << "\n";
//      std::cout << "w2[0,1]=" << W2(0,1) << "\n";
//      std::cout << "w2[0,2]=" << W2(0,2) << "\n";

//      std::cout << "w2[1,0]=" << W2(1,0) << "\n";
//      std::cout << "w2[1,1]=" << W2(1,1) << "\n";
//      std::cout << "w2[1,2]=" << W2(1,2) << "\n";

//      std::cout << "w2[2,0]=" << W2(2,0) << "\n";
//      std::cout << "w2[2,1]=" << W2(2,1) << "\n";
//      std::cout << "w2[2,2]=" << W2(2,2) << "\n";

//	std::cout << "b2\n";
//	myPrintMat(b2_d, n_classes, 1, 3, 1);
//      std::cout << "b2[0]=" << b2(0,0) << "\n";
//      std::cout << "b2[1]=" << b2(1,0) << "\n";
//      std::cout << "b2[2]=" << b2(2,0) << "\n";
   }

   void forward(const arma::mat& X){
       const double* x_ptr = X.memptr();
       std::cout << "Forward: x[0]=" << x_ptr[0] << "\n";
       if (X.n_cols!=n_batch){
         std::cout << "nngpu forward incorrect x_cols " << X.n_cols << "\n";
         return;
       }

       cudaMemcpy(Xd, x_ptr, sizeof(double)*n_batch*n_feats, cudaMemcpyHostToDevice);

   }

   void backward(const arma::mat& X, const arma::mat& Y, double lr, double reg){

   }
};

#endif
