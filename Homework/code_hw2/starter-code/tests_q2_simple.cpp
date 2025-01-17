#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>

#include "tests_q2.h"

const unsigned int kSizeMaskTest = 1;
const unsigned int kStartBitTest = 0;

std::vector<uint> computeBlockHistograms(const std::vector<uint>& keys,
        uint numBlocks,
        uint numBuckets, uint numBits,
        uint startBit, uint blockSize);

std::vector<uint> reduceLocalHistoToGlobal(const std::vector<uint>&
        blockHistograms,
        uint numBlocks, uint numBuckets);

std::vector<uint> scanGlobalHisto(const std::vector<uint>& globalHisto,
                                  uint numBuckets);

std::vector<uint> computeBlockExScanFromGlobalHisto(uint numBuckets,
        uint numBlocks,
        const std::vector<uint>& globalHistoExScan,
        const std::vector<uint>& blockHistograms);

void populateOutputFromBlockExScan(const std::vector<uint>& blockExScan,
                                   uint numBlocks,
                                   uint numBuckets, uint startBit,
                                   uint numBits, uint blockSize, const std::vector<uint>& keys,
                                   std::vector<uint>& sorted);


void WriteVectorToFile(const std::string& filename, std::vector<uint>& v) {
    std::ofstream outfile(filename.c_str());

    if(!outfile.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
    }

    for(uint i = 0; i < v.size(); ++i) {
        outfile << v[i] << std::endl;
    }

    outfile.close();
}

uint StringToUint(const std::string& line) {
    std::stringstream buffer;
    uint res;
    buffer << line;
    buffer >> res;
    return res;
}

std::vector<uint> ReadVectorFromFile(const std::string& filename) {
    std::ifstream infile(filename.c_str());

    if(!infile) {
        std::cerr << "Failed to load the file." << std::endl;
    }

    std::vector<uint> res;
    std::string line;

    while(true) {
        getline(infile, line);

        if(infile.fail()) {
            break;
        }

        res.push_back(StringToUint(line));
    }

    return res;
}

void Test1() {
    std::vector<uint> input = ReadVectorFromFile("test_files_simple/input");
    std::vector<uint> expected_output =
        ReadVectorFromFile("test_files_simple/blockhistograms");

        for(int i = 0; i < input.size(); i++){std::cout<<input[i] << ",";}
        std::cout<<"\n";
        for(int i = 0; i < input.size(); i++){std::cout<<expected_output[i] << ",";}

    uint blockSize = 2;
    uint numBlocks = 4;
    uint numBuckets = 2;
    std::vector<uint> blockHistograms = computeBlockHistograms(input, numBlocks,
                                        numBuckets, kSizeMaskTest, kStartBitTest, blockSize);
    bool success = true;
    EXPECT_VECTOR_EQ(expected_output, blockHistograms, &success);
    PRINT_SUCCESS(success);
}

void Test2() {
    std::vector<uint> blockHistograms =
        ReadVectorFromFile("test_files_simple/blockhistograms");
    std::vector<uint> input = ReadVectorFromFile("test_files_simple/input");
    std::vector<uint> expected_output =
        ReadVectorFromFile("test_files_simple/globalhisto");

    uint blockSize = 2;
    uint numBlocks = 4;
    uint numBuckets = 2;
    std::vector<uint> globalHisto = reduceLocalHistoToGlobal(blockHistograms,
                                    numBlocks, numBuckets);
    bool success = true;
    EXPECT_VECTOR_EQ(expected_output, globalHisto, &success);
    PRINT_SUCCESS(success);
}

void Test3() {
    std::vector<uint> globalHisto = ReadVectorFromFile("test_files_simple/globalhisto");
    std::vector<uint> expected_output =
        ReadVectorFromFile("test_files_simple/globalhistoexscan");

    uint numBuckets = 2;
    std::vector<uint> globalHistoExScan = scanGlobalHisto(globalHisto, numBuckets);
    bool success = true;
    EXPECT_VECTOR_EQ(expected_output, globalHistoExScan, &success);
    PRINT_SUCCESS(success);
}

void Test4() {
    std::vector<uint> input = ReadVectorFromFile("test_files_simple/input");
    std::vector<uint> expected_output =
        ReadVectorFromFile("test_files_simple/blockexscan");

    uint blockSize = 2;
    uint numBlocks = 4;
    uint numBuckets = 2;
    std::vector<uint> globalHistoExScan =
        ReadVectorFromFile("test_files_simple/globalhistoexscan");
    std::vector<uint> blockHistograms =
        ReadVectorFromFile("test_files_simple/blockhistograms");
    std::vector<uint> blockExScan = computeBlockExScanFromGlobalHisto(numBuckets,
                                    numBlocks, globalHistoExScan, blockHistograms);
    bool success = true;
    EXPECT_VECTOR_EQ(expected_output, blockExScan, &success);
    PRINT_SUCCESS(success);
}

void Test5() {
    std::vector<uint> blockExScan = ReadVectorFromFile("test_files_simple/blockexscan");
    std::vector<uint> input = ReadVectorFromFile("test_files_simple/input");
    std::vector<uint> expected_output = ReadVectorFromFile("test_files_simple/sorted");

    uint blockSize = 2;
    uint numBlocks = 4;
    uint numBuckets = 2;
    std::vector<uint> sorted(input.size());
    populateOutputFromBlockExScan(blockExScan, numBlocks, numBuckets, kStartBitTest,
                                  kSizeMaskTest, blockSize, input, sorted);

    bool success = true;
    EXPECT_VECTOR_EQ(expected_output, sorted, &success);
    PRINT_SUCCESS(success);
}
