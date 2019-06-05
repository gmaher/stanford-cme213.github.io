#include "neural_network.h"

#include <armadillo>
#include "utils/common.h"
#include "gpu_func.h"
#include "mpi.h"
#include "iomanip"
#include <cmath>
#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

double norms(NeuralNetwork& nn) {
    double norm_sum = 0;

    for(int i = 0; i < nn.num_layers; ++i)  {
        norm_sum += arma::accu(arma::square(nn.W[i]));
    }

    return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork& nn, int iter) {
    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    nn.W[0].save(s.str(), arma::raw_ascii);
    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    nn.W[1].save(t.str(), arma::raw_ascii);
    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    nn.b[0].save(u.str(), arma::raw_ascii);
    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork& nn, int iter,
                        std::ofstream& error_file) {
    arma::mat A, B, C, D;

    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    A.load(s.str(), arma::raw_ascii);
    double max_errW0 = arma::norm(nn.W[0]-A, "inf")/arma::norm(A, "inf");
    double L2_errW0  = arma::norm(nn.W[0]-A,2)/arma::norm(A,2);

    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    B.load(t.str(), arma::raw_ascii);
    double max_errW1 = arma::norm(nn.W[1]-B, "inf")/arma::norm(B, "inf");
    double L2_errW1  = arma::norm(nn.W[1]-B,2)/arma::norm(B,2);

    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    C.load(u.str(), arma::raw_ascii);
    double max_errb0 = arma::norm(nn.b[0]-C, "inf")/arma::norm(C, "inf");
    double L2_errb0  = arma::norm(nn.b[0]-C,2)/arma::norm(C,2);

    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    D.load(v.str(), arma::raw_ascii);
    double max_errb1 = arma::norm(nn.b[1]-D, "inf")/arma::norm(D, "inf");
    double L2_errb1  = arma::norm(nn.b[1]-D,2)/arma::norm(D,2);

    int ow = 15;

    if(iter == 0) {
        error_file << std::left<< std::setw(ow) << "Iteration" << std::left<< std::setw(
                       ow) << "Max Err W0" << std::left << std::setw(ow) << "Max Err W1"
                   << std::left<< std::setw(ow) << "Max Err b0" << std::left<< std::setw(
                       ow) << "Max Err b1" << std::left << std::setw(ow) << "L2 Err W0" << std::left
                   << std::setw(ow) << "L2 Err W1" << std::left<< std::setw(
                       ow) << "L2 Err b0" << std::left<< std::setw(ow) << "L2 Err b1" << "\n";
    }

    error_file << std::left << std::setw(ow) << iter << std::left << std::setw(
                   ow) << max_errW0 << std::left << std::setw(ow) << max_errW1 <<
               std::left << std::setw(ow) << max_errb0 << std::left << std::setw(
                   ow) << max_errb1 << std::left<< std::setw(ow) << L2_errW0 << std::left <<
               std::setw(ow) << L2_errW1 << std::left << std::setw(ow) << L2_errb0 <<
               std::left<< std::setw(ow) << L2_errb1 << "\n";

}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache) {
    cache.z.resize(2);
    cache.a.resize(2);

    // std::cout << W[0].n_rows << "\n";tw
    assert(X.n_rows == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_cols;

    arma::mat z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
    cache.z[0] = z1;

    arma::mat a1;
    sigmoid(z1, a1);
    cache.a[0] = a1;

    assert(a1.n_rows == nn.W[1].n_cols);
    arma::mat z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
    cache.z[1] = z2;

    arma::mat a2;
    softmax(z2, a2);
    cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::mat& y, double reg,
              const struct cache& bpcache, struct grads& bpgrads) {
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int N = y.n_cols;

    // std::cout << "backprop " << bpcache.yc << "\n";
    arma::mat diff = (1.0 / N) * (bpcache.yc - y);
    bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
    bpgrads.db[1] = arma::sum(diff, 1);
    arma::mat da1 = nn.W[1].t() * diff;

    // printf("diff[0,0]=%f\n",diff(0,0)*N);
    // printf("diff[1,0]=%f\n",diff(1,0)*N);
    // printf("diff[2,0]=%f\n",diff(2,0)*N);
    // printf("diff[0,1]=%f\n",diff(0,1)*N);
    // printf("diff[0,2]=%f\n",diff(0,2)*N);

    // for (int i = 0; i < 10; i++){
    //   printf("diff[0,%u]=%f\n",i,diff(0,i)*N);
    // }


    arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

    // printf("\nda1[0,0]=%f\n",da1(0,0)*N);
    // printf("da1[1,0]=%f\n",da1(1,0)*N);
    // printf("da1[2,0]=%f\n",da1(2,0)*N);
    // printf("da1[0,1]=%f\n",da1(0,1)*N);
    // printf("da1[0,2]=%f\n",da1(0,2)*N);
    // printf("da1[2,1]=%f\n",da1(2,1)*N);
    // printf("da1[2,2]=%f\n",da1(2,2)*N);

    // printf("\ndz1[0,0]=%f\n",dz1(0,0)*N);
    // printf("dz1[1,0]=%f\n",dz1(1,0)*N);
    // printf("dz1[2,0]=%f\n",dz1(2,0)*N);
    // printf("dz1[0,1]=%f\n",dz1(0,1)*N);
    // printf("dz1[0,2]=%f\n",dz1(0,2)*N);
    // printf("dz1[2,1]=%f\n",dz1(2,1)*N);
    // printf("dz1[2,2]=%f\n",dz1(2,2)*N);


    bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
    bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss(NeuralNetwork& nn, const arma::mat& yc, const arma::mat& y,
            double reg) {
    int N = yc.n_cols;
    double ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

    double data_loss = ce_sum / N;
    double reg_loss = 0.5 * reg * norms(nn);
    double loss = data_loss + reg_loss;
    // std::cout << "Loss: " << loss << "\n";
    return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::mat& X, arma::rowvec& label) {
    struct cache fcache;
    feedforward(nn, X, fcache);
    label.set_size(X.n_cols);

    for(int i = 0; i < X.n_cols; ++i) {
        arma::uword row;
        fcache.yc.col(i).max(row);
        label(i) = row;
    }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
             double reg, struct grads& numgrads) {
    double h = 0.00001;
    struct cache numcache;
    numgrads.dW.resize(nn.num_layers);
    numgrads.db.resize(nn.num_layers);

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

        for(int j = 0; j < nn.W[i].n_rows; ++j) {
            for(int k = 0; k < nn.W[i].n_cols; ++k) {
                double oldval = nn.W[i](j,k);
                nn.W[i](j, k) = oldval + h;
                feedforward(nn, X, numcache);
                double fxph = loss(nn, numcache.yc, y, reg);
                nn.W[i](j, k) = oldval - h;
                feedforward(nn, X, numcache);
                double fxnh = loss(nn, numcache.yc, y, reg);
                numgrads.dW[i](j, k) = (fxph - fxnh) / (2*h);
                nn.W[i](j, k) = oldval;
            }
        }
    }

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

        for(int j = 0; j < nn.b[i].size(); ++j) {
            double oldval = nn.b[i](j);
            nn.b[i](j) = oldval + h;
            feedforward(nn, X, numcache);
            double fxph = loss(nn, numcache.yc, y, reg);
            nn.b[i](j) = oldval - h;
            feedforward(nn, X, numcache);
            double fxnh = loss(nn, numcache.yc, y, reg);
            numgrads.db[i](j) = (fxph - fxnh) / (2*h);
            nn.b[i](j) = oldval;
        }
    }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
           double learning_rate, double reg,
           const int epochs, const int batch_size, bool grad_check, int print_every,
           int debug) {
    int N = X.n_cols;
    int iter = 0;
    int print_flag = 0;

    for(int epoch = 0 ; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches-1; ++batch) {
            int last_col = std::min((batch + 1)*batch_size-1, N-1);
            arma::mat X_batch = X.cols(batch * batch_size, last_col);
            arma::mat y_batch = y.cols(batch * batch_size, last_col);

            struct cache bpcache;
            feedforward(nn, X_batch, bpcache);

            struct grads bpgrads;
            backprop(nn, y_batch, reg, bpcache, bpgrads);

            if(print_every > 0 && iter % print_every == 0) {
                if(grad_check) {
                    struct grads numgrads;
                    numgrad(nn, X_batch, y_batch, reg, numgrads);
                    assert(gradcheck(numgrads, bpgrads));
                }

                std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" <<
                          epochs << " = " << loss(nn, bpcache.yc, y_batch, reg) << "\n";
            }

            // Gradient descent step
            for(int i = 0; i < nn.W.size(); ++i) {
                nn.W[i] -= learning_rate * bpgrads.dW[i];
            }

            for(int i = 0; i < nn.b.size(); ++i) {
                nn.b[i] -= learning_rate * bpgrads.db[i];
            }

            /* Debug routine runs only when debug flag is set. If print_every is zero, it saves
               for the first batch of each epoch to avoid saving too many large files.
               Note that for the first time, you have to run debug and serial modes together.
               This will run the following function and write out files to CPUmats folder.
               In the later runs (with same parameters), you can use just the debug flag to
               output diff b/w CPU and GPU without running CPU version */
            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            if(debug && print_flag) {
                write_cpudata_tofile(nn, iter);
            }

            iter++;
        }
    }
}

/*
 * TODO
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug) {

    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    int N = (rank == 0)?X.n_cols:0;
    int M = X.n_rows;
    int N_class = y.n_rows;
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;

    int proc_batch_size = batch_size/num_procs;
    NeuralNetworkGPU nn_gpu(M,N_class,nn.H[1],batch_size);

    if (rank == 0){
      std::cout << "num procs=" << num_procs << "\n";
      std::cout << "num cols X=" << N << "\n";
      std::cout << "num rows X=" << M << "\n";
      std::cout << "num classes Y=" << N_class << "\n";

      nn_gpu.set_weights(nn.W[0], nn.b[0], nn.W[1], nn.b[1]);
    }
    std::cout << "hello from rank " << rank << "\n";
    checkCudaErrors(cudaSetDevice(rank));

    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in NeuralNetwork &nn of rank 0 before returning from the function. */

    /* iter is a variable used to manage debugging. It increments in the inner loop
       and therefore goes from 0 to epochs*num_batches */

    /*
    1* broadcast x,y data to all procs
    2* adjust loop based on rank
    3* compute gradients
    4* all reduce gradients
    5* rank 0 compute new parameters
    6* broadcast new parameters
    7* set new parameters
    */

    //******************Broadcast data
    double* X_data_ptr_loc;
    double* Y_data_ptr_loc;

    X_data_ptr_loc = (double*)malloc(M*batch_size*sizeof(double));
    Y_data_ptr_loc = (double*)malloc(N_class*batch_size*sizeof(double));

    int iter = 0;

    for(int epoch = 0; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches-1; ++batch) {
          if (rank == 0){
            std::cout << "Par train iteration " << batch << "\n";
          }
            /*
             * Possible Implementation:
             * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
             * 2. compute each sub-batch of images' contribution to network coefficient updates
             * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
             * 4. update local network coefficient at each node
             */

             if (rank == 0){
               int last_col = std::min((batch + 1)*batch_size-1, N-1);
               arma::mat X_batch = X.cols(batch * batch_size, last_col);
               arma::mat y_batch = y.cols(batch * batch_size, last_col);

               const double* x_ptr = X_batch.memptr();
               const double* y_ptr = y_batch.memptr();
               std::copy(x_ptr, x_ptr+M*batch_size, X_data_ptr_loc);
               std::copy(y_ptr, y_ptr+N_class*batch_size, Y_data_ptr_loc);
             }

             MPI_Bcast(X_data_ptr_loc, M*batch_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
             // MPI_Bcast(Y_data_ptr_loc, N*N_class, MPI_DOUBLE, 0, MPI_COMM_WORLD);

             if(rank == 0){
               nn_gpu.forward(X_batch);
               nn_gpu.backward(X_batch, y_batch, learning_rate, reg);
             }


            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            /* Following debug routine assumes that you have already updated the arma
               matrices in the NeuralNetwork nn.  */
            if(debug && rank == 0 && print_flag) {
                nn_gpu.get_weights(nn.W[0], nn.b[0], nn.W[1], nn.b[1]);
                write_diff_gpu_cpu(nn, iter, error_file);
            }

            iter++;
        }
    }

    //copy weights
    if (rank == 0){
      std::cout << "writing gpu weights to cpu\n";
      nn_gpu.get_weights(nn.W[0], nn.b[0], nn.W[1], nn.b[1]);
    }

    if (rank != 0){
      free(X_data_ptr_loc);
      //free(Y_data_ptr_loc);
    }
    error_file.close();
}

void parallel_test(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug) {

    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    int N = (rank == 0)?X.n_cols:0;
    int M = X.n_rows;
    int N_class = y.n_rows;
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;

    if (rank == 0){
      std::cout << "num procs=" << num_procs << "\n";
      std::cout << "num cols X=" << N << "\n";
      std::cout << "num rows X=" << M << "\n";
      std::cout << "num classes Y=" << N_class << "\n";
      NeuralNetworkGPU nn_gpu(M,N_class,nn.H[1],batch_size);
      nn_gpu.set_weights(nn.W[0], nn.b[0], nn.W[1], nn.b[1]);
      nn_gpu.forward(X.cols(0,batch_size-1));
      nn_gpu.backward(X.cols(0,batch_size-1), y.cols(0,batch_size-1),
      learning_rate, reg);

      struct cache fcache;
      feedforward(nn, X.cols(0,batch_size-1), fcache);

      struct grads bpgrads;
      backprop(nn, y.cols(0,batch_size-1), reg, fcache, bpgrads);

      double* z1;
      double* a1;
      double* z2;
      double* a2;
      z1  = (double*)malloc(nn.H[1]*batch_size*sizeof(double));
      a1  = (double*)malloc(nn.H[1]*batch_size*sizeof(double));
      z2  = (double*)malloc(N_class*batch_size*sizeof(double));
      a2  = (double*)malloc(N_class*batch_size*sizeof(double));

      cudaMemcpy(z1, nn_gpu.z1_d, nn.H[1]*batch_size*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(a1, nn_gpu.a1_d, nn.H[1]*batch_size*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(z2, nn_gpu.z2_d, N_class*batch_size*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(a2, nn_gpu.a2_d, N_class*batch_size*sizeof(double), cudaMemcpyDeviceToHost);

      arma::mat z1_d_mat = arma::mat(z1, nn.H[1], batch_size);
      arma::mat a1_d_mat = arma::mat(a1, nn.H[1], batch_size);
      arma::mat z2_d_mat = arma::mat(z2, N_class, batch_size);
      arma::mat a2_d_mat = arma::mat(a2, N_class, batch_size);

      printf("\nz1_d_mat[0,0]=%f\n", z1_d_mat(0,0));
      printf("a1_d_mat[0,0]=%f\n", a1_d_mat(0,0));
      printf("z2_d_mat[0,0]=%f\n", z2_d_mat(0,0));
      printf("a2_d_mat[0,0]=%f\n", a2_d_mat(0,0));

      arma::mat z1_h = fcache.z[0];
      arma::mat a1_h = fcache.a[0];
      arma::mat z2_h = fcache.z[1];
      arma::mat a2_h = fcache.a[1];

      printf("\nz1_h[0,0]=%f\n", z1_h(0,0));
      printf("a1_h[0,0]=%f\n", a1_h(0,0));
      printf("z2_h[0,0]=%f\n", z2_h(0,0));
      printf("a2_h[0,0]=%f\n", a2_h(0,0));

      //grads
      double* dw1;
      double* db1;
      double* dw2;
      double* db2;
      dw1  = (double*)malloc(nn.H[1]*M*sizeof(double));
      db1  = (double*)malloc(nn.H[1]*batch_size*sizeof(double));
      dw2  = (double*)malloc(N_class*nn.H[1]*sizeof(double));
      db2  = (double*)malloc(N_class*batch_size*sizeof(double));

      cudaMemcpy(dw1, nn_gpu.dW1, nn.H[1]*M*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(db1, nn_gpu.db1, nn.H[1]*batch_size*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(dw2, nn_gpu.dW2, N_class*nn.H[1]*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(db2, nn_gpu.db2, N_class*batch_size*sizeof(double), cudaMemcpyDeviceToHost);

      arma::mat dw1_d_mat = arma::mat(dw1, nn.H[1], M);
      arma::mat db1_d_mat = arma::mat(db1, nn.H[1], 1);
      arma::mat dw2_d_mat = arma::mat(dw2, N_class, nn.H[1]);
      arma::mat db2_d_mat = arma::mat(db2, N_class, 1);
      printf("\ndw1_d_mat[10,300]=%f\n", dw1_d_mat(10,300));
      printf("db1_d_mat[10,0]=%f\n", db1_d_mat(10,0));
      printf("dw2_d_mat[0,50]=%f\n", dw2_d_mat(0,50));
      printf("db2_d_mat[0,0]=%f\n", db2_d_mat(0,0));


      arma::mat dw1_h = bpgrads.dW[0];
      arma::mat db1_h = bpgrads.db[0];
      arma::mat dw2_h = bpgrads.dW[1];
      arma::mat db2_h = bpgrads.db[1];

      printf("\ndw1_h[10,300]=%f\n", dw1_h(10,300));
      printf("db1_h[10,0]=%f\n", db1_h(10,0));
      printf("dw2_h[0,50]=%f\n", dw2_h(0,50));
      printf("db2_h[0,0]=%f\n", db2_h(0,0));

    }
}
