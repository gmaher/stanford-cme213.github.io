#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
/* TODO: Make Matrix a pure abstract class with the
 * public method:
 *     std::string repr()
 */
class Matrix {
public:
 virtual std::string repr() = 0;
};

/* TODO: Modify the following function so that it
 * inherits from the Matrix class */
class SparseMatrix : public Matrix{
 public:
  std::string repr() {
    return "sparse";
  }
};

/* TODO: Modify the following function so that it
 * inherits from the Matrix class */
class ToeplitzMatrix : public Matrix{
 public:
  std::string repr() {
    return "toeplitz";
  }
};

/* TODO: This function should accept a vector of Matrices and call the repr
 * function on each matrix, printing the result to standard out.
 */
void PrintRepr(std::vector<Matrix*>& matVec){
  std::for_each(matVec.begin(), matVec.end(),
    [](Matrix* A){std::cout << A->repr() << "\n";});
};

/* TODO: Your main method should fill a vector with an instance of SparseMatrix
 * and an instance of ToeplitzMatrix and pass the resulting vector
 * to the PrintRepr function.
 */
int main() {

  auto matVec = std::vector<Matrix*>();
  SparseMatrix A;
  ToeplitzMatrix B;
  matVec.push_back(&A);
  matVec.push_back(&B);

  PrintRepr(matVec);

  return 0;
}
