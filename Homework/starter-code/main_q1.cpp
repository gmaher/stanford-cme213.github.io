#include "matrix.hpp"

int main(){

  MatrixSymmetric<unsigned> A(4);

  A(0,0) = 0;
  A(0,1) = 1;
  A(0,2) = 3;
  A(0,3) = 4;
  A(1,1) = 5;
  A(1,2) = 6;
  A(1,3) = 7;
  A(2,2) = 8;
  A(2,3) = 9;
  A(3,3) = 10;

  std::cout << A;

  std::cout << "non zeros: " << A.l0_norm() << "\n";
  return 0;
}
