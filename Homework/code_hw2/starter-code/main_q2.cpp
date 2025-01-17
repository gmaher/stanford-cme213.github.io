#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include <assert.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <cstdio>

#include "tests_q2.h"

typedef unsigned int uint;

const uint kSizeTestVector = 4000000;
const uint kSizeMask =
    16; // must be a divider of 32 for this program to work correctly
const uint kRandMax = 1 << 31;
const uint kNumBitsUint = 32;

/* Function: computeBlockHistograms
 * --------------------------------
 * Splits keys into numBlocks and computes an histogram with numBuckets buckets
 * Remember that numBuckets and numBits are related; same for blockSize and numBlocks.
 * Should work in parallel.
 */
std::vector<uint> computeBlockHistograms(const std::vector<uint>& keys,
        uint numBlocks, uint numBuckets,
        uint numBits, uint startBit, uint blockSize) {
    std::vector<uint> blockHistograms(numBlocks * numBuckets, 0);

    #pragma omp parallel for
    for (int i = 0; i < numBlocks; i++){
      for(int b=0; b<blockSize; b++){
        int index=i*blockSize+b;
        if(index >= keys.size()) break;

        int bucket = 0;
        for (int j = 0; j < numBits; j++){
          if( (keys[index] >> (startBit+j)) & 1){
            bucket += pow(2,j);
          }
        }

        blockHistograms[i*numBuckets + bucket]+=1;
      }
    }

    return blockHistograms;
}

/* Function: reduceLocalHistoToGlobal
 * ----------------------------------
 * Takes as input the local histogram of size numBuckets * numBlocks and "merges"
 * them into a global histogram of size numBuckets.
 */
std::vector<uint> reduceLocalHistoToGlobal(const std::vector<uint>&
        blockHistograms, uint numBlocks, uint numBuckets) {
    std::vector<uint> globalHisto(numBuckets, 0);

    for(int b = 0; b < numBlocks; b++){
      for(int i = 0; i < numBuckets; i++){
        int index = b*numBuckets+i;
        if(index >= blockHistograms.size()) break;
        globalHisto[i] += blockHistograms[index];
      }
    }

    return globalHisto;
}

/* Function: scanGlobalHisto
 * -------------------------
 * This function should simply scan the global histogram.
 */
std::vector<uint> scanGlobalHisto(const std::vector<uint>& globalHisto,
                                  uint numBuckets) {
    std::vector<uint> globalHistoExScan(numBuckets, 0);

    int sum = 0;
    for (int i = 0; i < numBuckets; i++){
      globalHistoExScan[i] = sum;
      sum += globalHisto[i];
    }

    return globalHistoExScan;
}

/* Function: computeBlockExScanFromGlobalHisto
 * -------------------------------------------
 * Takes as input the globalHistoExScan that contains the global histogram after the scan
 * and the local histogram in blockHistograms. Returns a local histogram that will be used
 * to populate the sorted array.
 */
std::vector<uint> computeBlockExScanFromGlobalHisto(uint numBuckets,
        uint numBlocks,
        const std::vector<uint>& globalHistoExScan,
        const std::vector<uint>& blockHistograms) {
    std::vector<uint> blockExScan(numBuckets * numBlocks, 0);

    for (int i = 0; i < globalHistoExScan.size(); i++){
      blockExScan[i] = globalHistoExScan[i];
    }

    for (int b = 1; b < numBlocks; b++){
      for (int i = 0; i < numBuckets; i++){
        int b_prev = (b-1)*numBuckets+i;
        int b_next = b*numBuckets+i;
        if(b_next >= blockExScan.size()) break;
        blockExScan[b_next] += blockExScan[b_prev]+blockHistograms[b_prev];
      }
    }

    return blockExScan;
}

/* Function: populateOutputFromBlockExScan
 * ---------------------------------------
 * Takes as input the blockExScan produced by the splitting of the global histogram
 * into blocks and populates the vector sorted.
 */
void populateOutputFromBlockExScan(const std::vector<uint>& blockExScan,
                                   uint numBlocks, uint numBuckets, uint startBit,
                                   uint numBits, uint blockSize, const std::vector<uint>& keys,
                                   std::vector<uint>& sorted) {

   #pragma omp parallel for
   for (int i = 0; i < numBlocks; i++){

       std::vector<uint> offsets(numBuckets,0);
       int block_offset = i*blockSize;
       for (int b=0; b < blockSize; b++){
           int index = block_offset+b;

           if(index >= keys.size()) break;

           int bucket = 0;
           for (int j = 0; j < numBits; j++){
               if( (keys[index] >> (startBit+j)) & 1){
                 bucket += pow(2,j);
               }
           }

           int b_index = i*numBuckets+bucket;
           if(b_index >= blockExScan.size()) break;

           int key_index = offsets[bucket]+blockExScan[b_index];

           sorted[key_index] = keys[index];
           offsets[bucket] += 1;
       }
   }


}

/* Function: radixSortParallelPass
 * -------------------------------
 * A pass of radixSort on numBits starting after startBit.
 */
void radixSortParallelPass(std::vector<uint>& keys, std::vector<uint>& sorted,
                           uint numBits, uint startBit,
                           uint blockSize) {
    uint numBuckets = 1 << numBits;
    // Choose numBlocks so that numBlocks * blockSize is always greater than keys.size().
    uint numBlocks = (keys.size() + blockSize - 1) / blockSize;

    // go over each block and compute its local histogram
    std::vector<uint> blockHistograms = computeBlockHistograms(keys, numBlocks,
                                        numBuckets, numBits, startBit, blockSize);

    // first reduce all the local histograms into a global one
    std::vector<uint> globalHisto = reduceLocalHistoToGlobal(blockHistograms,
                                    numBlocks, numBuckets);

    // now we scan this global histogram
    std::vector<uint> globalHistoExScan = scanGlobalHisto(globalHisto, numBuckets);

    // now we do a local histogram in each block and add in the global value to get global position
    std::vector<uint> blockExScan = computeBlockExScanFromGlobalHisto(numBuckets,
                                    numBlocks, globalHistoExScan, blockHistograms);

    // populate the sorted vector
    populateOutputFromBlockExScan(blockExScan, numBlocks, numBuckets, startBit,
                                  numBits, blockSize, keys, sorted);
}

int radixSortParallel(std::vector<uint>& keys, std::vector<uint>& keys_tmp,
                      uint numBits, uint numBlocks) {

    uint blockSize = keys.size()/numBlocks;
    for(uint startBit = 0; startBit < kNumBitsUint; startBit += 2 * numBits) {
        radixSortParallelPass(keys, keys_tmp, numBits, startBit, blockSize);
        radixSortParallelPass(keys_tmp, keys, numBits, startBit + numBits,
                              blockSize);
    }

    return 0;
}

void radixSortSerialPass(std::vector<uint>& keys, std::vector<uint>& keys_radix,
                         uint startBit, uint numBits) {
    uint numBuckets = 1 << numBits;
    uint mask = numBuckets - 1;

    //compute the frequency histogram
    std::vector<uint> histogramRadixFrequency(numBuckets);

    for(uint i = 0; i < keys.size(); ++i) {
        uint key = (keys[i] >> startBit) & mask;
        ++histogramRadixFrequency[key];
    }

    //now scan it
    std::vector<uint> exScanHisto(numBuckets, 0);

    for(uint i = 1; i < numBuckets; ++i) {
        exScanHisto[i] = exScanHisto[i - 1] + histogramRadixFrequency[i-1];
        histogramRadixFrequency[i - 1] = 0;
    }

    histogramRadixFrequency[numBuckets - 1] = 0;

    //now add the local to the global and scatter the result
    for(uint i = 0; i < keys.size(); ++i) {
        uint key = (keys[i] >> startBit) & mask;

        uint localOffset = histogramRadixFrequency[key]++;
        uint globalOffset = exScanHisto[key] + localOffset;

        keys_radix[globalOffset] = keys[i];
    }
}

int radixSortSerial(std::vector<uint>& keys, std::vector<uint>& keys_radix,
                    uint numBits) {
    assert(numBits <= 16);

    for(uint startBit = 0; startBit < 32; startBit += 2 * numBits) {
        radixSortSerialPass(keys, keys_radix, startBit, numBits);
        radixSortSerialPass(keys_radix, keys, startBit+numBits, numBits);
    }

    return 0;
}

void initializeRandomly(std::vector<uint>& keys) {
    std::default_random_engine generator;
    std::uniform_int_distribution<uint> distribution(0, kRandMax);

    for(uint i = 0; i < keys.size(); ++i) {
        keys[i] = distribution(generator);
    }
}

int main() {
    Test1();
    Test2();
    Test3();
    Test4();
    Test5();

    // Initialize Variables
    std::vector<uint> keys_stl(kSizeTestVector);
    initializeRandomly(keys_stl);
    std::vector<uint> keys_serial = keys_stl;
    std::vector<uint> keys_parallel = keys_stl;

    std::vector<uint> temp_keys(kSizeTestVector);

#ifdef QUESTION6
    std::vector<uint> keys_orig = keys_stl;

#endif

    // stl sort
    double startstl = omp_get_wtime();
    std::sort(keys_stl.begin(), keys_stl.end());
    double endstl = omp_get_wtime();

    // serial radix sort
    double startRadixSerial = omp_get_wtime();
    radixSortSerial(keys_serial, temp_keys, kSizeMask);
    double endRadixSerial = omp_get_wtime();

    bool success = true;
    EXPECT_VECTOR_EQ(keys_stl, keys_serial, &success);

    if(success) {
        std::cout << "Serial Radix Sort is correct" << std::endl;
    }

    // parallel radix sort
    double startRadixParallel = omp_get_wtime();
    radixSortParallel(keys_parallel, temp_keys, kSizeMask, 12);
    double endRadixParallel = omp_get_wtime();

    success = true;
    EXPECT_VECTOR_EQ(keys_stl, keys_parallel, &success);

    if(success) {
        std::cout << "Parallel Radix Sort is correct" << std::endl;
    }

    std::cout << "stl: " << endstl - startstl << std::endl;
    std::cout << "serial radix: " << endRadixSerial - startRadixSerial << std::endl;
    std::cout << "parallel radix: " << endRadixParallel - startRadixParallel <<
              std::endl;

#ifdef QUESTION6
    std::vector<uint> jNumBlock = {1,2,4,8,12,16,24,32,40,48};
    printf("Threads Blocks / Timing\n ");
    for(auto jNum : jNumBlock) {
      printf("%8d", jNum);
    }
    printf("\n");
    success = true;
    for(auto n_threads : jNumBlock) {
      omp_set_num_threads(n_threads);
      printf("%4d ", n_threads);
      for(auto jNum : jNumBlock) {
        keys_parallel = keys_orig;
        double startRadixParallel = omp_get_wtime();

        radixSortParallel(keys_parallel, temp_keys, kSizeMask, jNum);

        double endRadixParallel = omp_get_wtime();
        EXPECT_VECTOR_EQ(keys_stl, keys_parallel, &success);
        printf("%8.3f", endRadixParallel - startRadixParallel);
      }
      printf("\n");
    }
    if(success) {
    std::cout << "Benchmark runs: PASS" << std::endl;
    } else {
    std::cout << "Benchmark runs: FAIL" << std::endl;
    }

#endif

    return 0;
}
