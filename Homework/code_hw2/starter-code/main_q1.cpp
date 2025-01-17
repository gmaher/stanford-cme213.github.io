#include <iostream>
#include <random>
#include <vector>

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <sys/time.h>

#include "tests_q1.h"

typedef unsigned int uint;
const uint kMaxInt = 100;
const uint kSize = 30000000;

std::vector<uint> serialSum(const std::vector<uint>& v) {
    std::vector<uint> sums(2);
    for (std::vector<uint>::const_iterator it = v.begin();
          it != v.end();
            ++it){

      if ((*it)%2 == 0){
        sums[0] += *it;
      }else{
        sums[1] += *it;
      }

    }
    return sums;
}

std::vector<uint> parallelSum(const std::vector<uint>& v) {
    std::vector<uint> sums(2);

    #pragma omp for
    for (int i = 0; i < v.size(); i++){

        if (v[i]%2 == 0){
          sums[0] += v[i];
        }else{
          sums[1] += v[i];
        }
    }

    return sums;
}

std::vector<uint> initializeRandomly(const uint size, const uint max_int) {
    std::vector<uint> res(size);
    std::default_random_engine generator;
    std::uniform_int_distribution<uint> distribution(0, max_int);

    for(uint i = 0; i < size; ++i) {
        res[i] = distribution(generator);
    }

    return res;
}

int main() {

    setenv("OMP_NUM_THREADS", "10", 0);

    // You can uncomment the line below to make your own simple tests
    //std::vector<uint> v = ReadVectorFromFile("vec");
    std::vector<uint> v = initializeRandomly(kSize, kMaxInt);

    std::cout << "Parallel" << std::endl;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    std::vector<uint> sums = parallelSum(v);
    std::cout << "Sum Even: " << sums[0] << std::endl;
    std::cout << "Sum Odd: " << sums[1] << std::endl;
    gettimeofday(&end, NULL);
    double delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec -
                    start.tv_usec) / 1.e6;
    std::cout << "Time: " << delta << std::endl;

    std::cout << "Serial" << std::endl;
    gettimeofday(&start, NULL);
    std::vector<uint> sumsSer = serialSum(v);
    std::cout << "Sum Even: " << sumsSer[0] << std::endl;
    std::cout << "Sum Odd: " << sumsSer[1] << std::endl;
    gettimeofday(&end, NULL);
    delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec -
             start.tv_usec) / 1.e6;
    std::cout << "Time: " << delta << std::endl;

    bool success = true;
    EXPECT_VECTOR_EQ(sums, sumsSer, &success);
    PRINT_SUCCESS(success);

    return 0;
}
