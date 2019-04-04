#include "matrix.hpp"
#include <assert.h>

int main(){

  unsigned N=5;

  MatrixSymmetric<unsigned> A(N);

  for (unsigned j = 0; j < N; j++){
    for (unsigned i = 0; i < N; i++){
      if (i > j){break;}

      A(i,j) = i+j;
    }
  }


  std::cout << A;

  std::cout << "non zeros: " << A.l0_norm() << "\n";

  std::cout << "testing entry\n";
  for (unsigned j = 0; j < N; j++){
    for (unsigned i = 0; i < N; i++){
      assert(A(i,j) == i+j);
    }
  }
  
  std::cout << "testing symmetry\n";
  for (unsigned j = 0; j < N; j++){
    for (unsigned i = 0; i < N; i++){
      assert(A(i,j) == A(j,i));
    }
  }

  std::cout << "testing l0 norm\n";
  assert(A.l0_norm() == N*N-1);

  return 0;
}
