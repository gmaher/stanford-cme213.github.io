#ifndef UTILS_TWO_LAYER_NET_H_
#define UTILS_TWO_LAYER_NET_H_

#include <armadillo>
#include <cmath>
#include <iostream>

class NeuralNetwork {
public:
    const int num_layers = 2;
    // H[i] is the number of neurons in layer i (where i=0 implies input layer)
    std::vector<int> H;
    // Weights of the neural network
    // W[i] are the weights of the i^th layer
    std::vector<arma::mat> W;
    // Biases of the neural network
    // b[i] is the row vector biases of the i^th layer
    std::vector<arma::colvec> b;

    NeuralNetwork(std::vector<int> _H) {
        W.resize(num_layers);
        b.resize(num_layers);
        H = _H;

        for(int i = 0; i < num_layers; i++) {
            arma::arma_rng::set_seed(arma::arma_rng::seed_type(i));
            W[i] = 0.01 * arma::randn(H[i+1], H[i]);
            b[i].zeros(H[i+1]);
        }
    }
};

void feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& bpcache);
double loss(NeuralNetwork& nn, const arma::mat& yc, const arma::mat& y,
            double reg);
void backprop(NeuralNetwork& nn, const arma::mat& y, double reg,
              const struct cache& bpcache, struct grads& bpgrads);
void numgrad(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
             double reg, struct grads& numgrads);
void train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
           double learning_rate, double reg = 0.0, const int epochs = 15,
           const int batch_size = 800, bool grad_check = false, int print_every = -1,
           int debug = 0);
void predict(NeuralNetwork& nn, const arma::mat& X, arma::rowvec& label);

void parallel_train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg = 0.0, const int epochs = 15,
                    const int batch_size = 800, bool grad_check = false, int print_every = -1,
                    int debug = 0);

class NeuralNetworkGPU {
public:

  int n_feats;
  int n_classes;
  int n_batch;

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
    NeuralNetworkGPU(NeuralNetwork nn, int x_size, int y_size, int batch_size) {
      n_feats = x_size;
      n_classes = y_size;
      n_batch = batch_size;

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


#endif
