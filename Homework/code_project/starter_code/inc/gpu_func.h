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

int mySigmoid(double* X, double* S, int M, int N);
int myHadamard(double* X, double* Y, double* H, int M, int N);
int myTranspose(double* X, double* Xt, int M, int N);
int myMatAdd(double* X, double* Y, double* Z, int M, int N, double alpha);
int mySoftmax(double* X, double* S, int M, int N);
int myRowSum(double* X, double* S, int M, int N, double alpha);
int myPrintMat(double* X, int M, int N, int m, int n);

class NeuralNetworkGPU {
public:
  int n_feats;
  int n_classes;
  int n_batch;
  int n_hidden;
  int num_procs;

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

  double* a1t_d;
  double* d2;
  double* dW2;
  double* db2;
  double* W2t_d;
  double* d1;
  double* a12_d;
  double* Xt_d;
  double* dW1;
  double* db1;

  double* dW1_h;
  double* db1_h;
  double* dW2_h;
  double* db2_h;

  double* dW1_h_2;
  double* db1_h_2;
  double* dW2_h_2;
  double* db2_h_2;
int rank;
   NeuralNetworkGPU(int x_size, int y_size, int hidden_size, int batch_size,
    int num_procs_, int rank_) {
       n_feats   = x_size;
       n_classes = y_size;
       n_batch   = batch_size;
       n_hidden  = hidden_size;
       num_procs = num_procs_;
       rank = rank_;
	printf("feats=%u, classes=%u, batch=%u, hidden=%u",n_feats, n_classes, n_batch, n_hidden);
       cudaMalloc((void**)&Xd, sizeof(double)*x_size*batch_size);
       cudaMalloc((void**)&Yd, sizeof(double)*y_size*batch_size);

       int h1 = n_hidden;
       int w1 = n_feats;
       cudaMalloc((void**)&W1_d, sizeof(double)*h1*w1);
       cudaMalloc((void**)&b1_d, sizeof(double)*h1*batch_size);
       cudaMalloc((void**)&z1_d, sizeof(double)*h1*batch_size);
       cudaMalloc((void**)&a1_d, sizeof(double)*h1*batch_size);
       std::cout << "layer 1 h=" << h1 << ", w=" << w1 << "\n";

       int h2 = n_classes;
       int w2 = n_hidden;
       cudaMalloc((void**)&W2_d, sizeof(double)*h2*w2);
       cudaMalloc((void**)&b2_d, sizeof(double)*h2*batch_size);
       cudaMalloc((void**)&z2_d, sizeof(double)*h2*batch_size);
       cudaMalloc((void**)&a2_d, sizeof(double)*h2*batch_size);
       std::cout << "layer 2 h=" << h2 << ", w=" << w2 << "\n";

       cudaMalloc((void**)&d2, sizeof(double)*y_size*batch_size);
       cudaMalloc((void**)&a1t_d, sizeof(double)*h1*batch_size);
       cudaMalloc((void**)&dW2, sizeof(double)*y_size*h1);
       cudaMalloc((void**)&db2, sizeof(double)*y_size*batch_size);

       cudaMalloc((void**)&W2t_d, sizeof(double)*y_size*h1);
       cudaMalloc((void**)&d1, sizeof(double)*h1*batch_size);
       cudaMalloc((void**)&a12_d, sizeof(double)*h1*batch_size);
       cudaMalloc((void**)&Xt_d, sizeof(double)*x_size*batch_size);

       cudaMalloc((void**)&dW1, sizeof(double)*h1*n_feats);
       cudaMalloc((void**)&db1, sizeof(double)*h1*batch_size);

       dW1_h = (double*)malloc(sizeof(double)*h1*n_feats);
       db1_h = (double*)malloc(sizeof(double)*h1*n_batch);
       dW2_h = (double*)malloc(sizeof(double)*n_classes*h1);
       db2_h = (double*)malloc(sizeof(double)*n_classes*n_batch);

       dW1_h_2 = (double*)malloc(sizeof(double)*h1*n_feats);
       db1_h_2 = (double*)malloc(sizeof(double)*h1*n_batch);
       dW2_h_2 = (double*)malloc(sizeof(double)*n_classes*h1);
       db2_h_2 = (double*)malloc(sizeof(double)*n_classes*n_batch);


     }

   void set_weights(const arma::mat& W1, const arma::mat& b1, const arma::mat& W2, const arma::mat& b2){
    	const double* w1_ptr = W1.memptr();
    	arma::mat b1_rep = arma::repmat(b1, 1, n_batch);
    	const double* b1_ptr = b1_rep.memptr();
    	const double* w2_ptr = W2.memptr();
    	arma::mat b2_rep = arma::repmat(b2,1,n_batch);
    	const double* b2_ptr = b2_rep.memptr();

    	cudaMemcpy(W1_d, w1_ptr, sizeof(double)*n_hidden*n_feats, cudaMemcpyHostToDevice);
    	cudaMemcpy(b1_d, b1_ptr, sizeof(double)*n_hidden*n_batch, cudaMemcpyHostToDevice);
    	cudaMemcpy(W2_d, w2_ptr, sizeof(double)*n_classes*n_hidden, cudaMemcpyHostToDevice);
    	cudaMemcpy(b2_d, b2_ptr, sizeof(double)*n_classes*n_batch, cudaMemcpyHostToDevice);
   }

   void get_weights(arma::mat& W1, arma::mat& b1, arma::mat& W2, arma::mat& b2){
    	double* w1_ptr = W1.memptr();
      double* b1_ptr = b1.memptr();
    	double* w2_ptr = W2.memptr();
    	double* b2_ptr = b2.memptr();

    	cudaMemcpy(w1_ptr, W1_d, sizeof(double)*n_hidden*n_feats, cudaMemcpyDeviceToHost);
    	cudaMemcpy(b1_ptr, b1_d,  sizeof(double)*n_hidden, cudaMemcpyDeviceToHost);
    	cudaMemcpy(w2_ptr, W2_d,  sizeof(double)*n_classes*n_hidden, cudaMemcpyDeviceToHost);
    	cudaMemcpy(b2_ptr, b2_d, sizeof(double)*n_classes, cudaMemcpyDeviceToHost);
   }

   void forward(const arma::mat& X){
       const double* x_ptr = X.memptr();
       if (X.n_cols!=n_batch){
         std::cout << "nngpu forward incorrect x_cols " << X.n_cols << "\n";
         return;
       }

       cudaMemcpy(Xd, x_ptr, sizeof(double)*n_batch*n_feats, cudaMemcpyHostToDevice);

       double alpha1 = 1.0;
       double beta1 = 0.0;
       myGEMM(W1_d, Xd, z1_d, &alpha1, &beta1, n_hidden, n_batch, n_feats);
       myMatAdd(z1_d, b1_d, z1_d, n_hidden, n_batch, 1.0);
       mySigmoid(z1_d, a1_d, n_hidden, n_batch);

      myGEMM(W2_d, a1_d, z2_d, &alpha1, &beta1, n_classes, n_batch, n_hidden);
      myMatAdd(z2_d, b2_d, z2_d, n_classes, n_batch, 1.0);
      mySoftmax(z2_d, a2_d, n_classes, n_batch);

   }

   void backward(const arma::mat& X, const arma::mat& Y, double reg){
     const double* y_ptr = Y.memptr();
     cudaMemcpy(Yd, y_ptr, sizeof(double)*n_batch*n_classes, cudaMemcpyHostToDevice);
     myMatAdd(a2_d, Yd, d2, n_classes, n_batch, -1.0);

     myTranspose(a1_d, a1t_d, n_hidden, n_batch);

     double scale = 1.0/n_batch;
     cudaMemcpy(dW2, W2_d, sizeof(double)*n_classes*n_hidden, cudaMemcpyDeviceToDevice);
     myGEMM(d2,a1t_d,dW2, &scale, &reg, n_classes,n_hidden, n_batch);
     myRowSum(d2, db2, n_classes, n_batch, scale);

     myTranspose(W2_d, W2t_d, n_classes, n_hidden);
     //myPrintMat(W2_d, n_classes, n_hidden, 3, 3);
     //myPrintMat(W2t_d, n_hidden, n_classes, 3, 3);


     double alpha1 = 1.0;
     double beta1 = 0.0;
     myGEMM(W2t_d, d2, d1, &alpha1, &beta1, n_hidden, n_batch, n_classes);
     //myPrintMat(d2, n_classes, n_batch, 1, 10);

     //myPrintMat(d1, n_hidden, n_batch, 3, 3);

     myHadamard(a1_d,a1_d,a12_d,n_hidden, n_batch);
     myMatAdd(a1_d,a12_d,a12_d,n_hidden,n_batch,-1.0);
     myHadamard(d1,a12_d,d1,n_hidden,n_batch);

     //myPrintMat(d1, n_hidden, n_batch, 3, 3);

     myTranspose(Xd, Xt_d, n_feats, n_batch);
     cudaMemcpy(dW1, W1_d, sizeof(double)*n_hidden*n_feats, cudaMemcpyDeviceToDevice);
     myGEMM(d1,Xt_d,dW1,&scale,&reg,n_hidden,n_feats,n_batch);
     myRowSum(d1, db1, n_hidden, n_batch, scale);
   }

  void gradientToHost(){
    cudaMemcpy(dW1_h, dW1, sizeof(double)*n_hidden*n_feats, cudaMemcpyDeviceToHost);
    cudaMemcpy(db1_h, db1, sizeof(double)*n_hidden*n_batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(dW2_h, dW2, sizeof(double)*n_classes*n_hidden, cudaMemcpyDeviceToHost);
    cudaMemcpy(db2_h, db2, sizeof(double)*n_classes*n_batch, cudaMemcpyDeviceToHost);
    std::cout << rank << " db1_h " << db1_h[0] << "\n";
  }

  void gradientToDevice(){
    // cudaMemcpy(dW1, dW1_h_2, sizeof(double)*n_hidden*n_feats, cudaMemcpyHostToDevice);
    // cudaMemcpy(db1, db1_h_2, sizeof(double)*n_hidden*n_batch, cudaMemcpyHostToDevice);
    // cudaMemcpy(dW2, dW2_h_2, sizeof(double)*n_classes*n_hidden, cudaMemcpyHostToDevice);
    // cudaMemcpy(db2, db2_h_2, sizeof(double)*n_classes*n_batch, cudaMemcpyHostToDevice);
    std::cout << rank << " db2_h_2 " << db2_h_2[0] << "\n";
  }

  void gradientStep(double lr){
    myMatAdd(W1_d, dW1, W1_d, n_hidden, n_feats, -lr);
    myMatAdd(b1_d, db1, b1_d, n_hidden, n_batch, -lr);
    myMatAdd(W2_d, dW2, W2_d, n_classes, n_hidden, -lr);
    myMatAdd(b2_d, db2, b2_d, n_classes, n_batch, -lr);
  }
};

#endif
