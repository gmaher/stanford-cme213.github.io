#ifndef _MATRIX_HPP
#define _MATRIX_HPP

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <exception>

struct out_of_range_exception : std::exception {
  const char* what() const noexcept {return "Requested element out of range\n";}
};

template<typename T>
class Matrix
{
  public:

    virtual T& operator()(const unsigned& i, const unsigned& j) = 0;

    virtual unsigned l0_norm() = 0;

    virtual std::ostream& print(std::ostream& os) = 0;

  protected:
    unsigned n;
};

template<typename T>
class MatrixSymmetric : public Matrix<T>
{
  public:
    MatrixSymmetric<T>(const unsigned& _n){
      n=_n;
      unsigned size = (n*(n+1))/2;
      data = std::vector<T>(size);
    }

    T& operator()(const unsigned& i, const unsigned& j){
      if (i >= n || j >= n){
        throw out_of_range_exception();
      }
      if (i  <= j){
        return data[i*n - ((i-1)*i)/2 + j-i];
      }
      else{
        return data[j*n - ((j-1)*j)/2 + i-j];
      }

    }

    unsigned l0_norm(){
      unsigned count = 2*std::count_if(data.begin(), data.end(),
                [](T x){return (x>0);});

      auto range = std::vector<unsigned>(n);
      std::iota(range.begin(), range.end(),0);

      count = count - std::count_if(range.begin(), range.end(),
        [this](int i){return ((*this)(i,i) > 0);}
      );

      return count;
    }

    std::ostream& print(std::ostream& os){
      for (unsigned i = 0; i < n; i++){
        for (unsigned j = 0; j < n; j++){
          os << (*this)(i,j) << ",";
        }
        os << "\n";
      }
    }


  protected:
    unsigned n;
    std::vector<T> data;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, Matrix<T>& A){
  return A.print(os);
}

#endif /* MATRIX_HPP */
