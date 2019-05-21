#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
// You may include other thrust headers if necessary.

#include "test_macros.h"

// You will need to call these functors from
// thrust functions in the code do not create new ones

// returns true if the char is not a lowercase letter
struct isnot_lowercase_alpha : thrust::unary_function<unsigned char, bool>
{
    __host__ __device__
    bool operator()(const unsigned char& x) const
    {
      return islower(x);
    }
};

// convert an uppercase letter into a lowercase one
// do not use the builtin C function or anything from boost, etc.
struct upper_to_lower : thrust::unary_function<unsigned char, unsigned char>
{
    __host__ __device__
    unsigned char operator()(const unsigned char& x) const
    {
      return x+32;
    }
};

// apply a shift with appropriate wrapping
struct apply_shift : thrust::binary_function<unsigned char, int, unsigned char>
{
    const unsigned char* shifts;
    const int period;

    apply_shift(unsigned char* shifts_, int& period_): period(period_){
      shifts = shifts_;
    }

    __host__ __device__
    unsigned char operator()(const unsigned char& x, const int& loc){
      return x + *(shifts+(loc%period));
    }
};

// Returns a vector with the top 5 letter frequencies in text.
std::vector<double> getLetterFrequencyCpu(const std::vector<unsigned char> &text)
{
    std::vector<unsigned int> freq(256);

    for (unsigned char c : text)
        freq[tolower(c)]++;

    unsigned int sum_chars = std::accumulate(
        freq.begin() + 'a',
        freq.begin() + 'z' + 1,
        0,
        std::plus<unsigned int>()
    );

    std::vector<double> freq_alpha_lower;

    for (unsigned char c = 'a'; c <= 'z'; ++c)
        if (freq[c] > 0)
            freq_alpha_lower.push_back(freq[c] / static_cast<double>(sum_chars));

    // pick the 5 most commonly occurring letters
    std::sort(freq_alpha_lower.begin(), freq_alpha_lower.end(), std::greater<double>());
    freq_alpha_lower.resize(std::min(static_cast<int>(freq_alpha_lower.size()), 5));
    return freq_alpha_lower;
}

// Print the top 5 letter frequencies and them.
std::vector<double> getLetterFrequencyGpu(const thrust::device_vector<unsigned char> &text)
{
    std::vector<double> freq_alpha_lower;
    // WARNING: make sure you handle the case of not all letters appearing
    // in the text.

    thrust::device_vector<unsigned char> text_sorted(text.size());
    thrust::copy(text.begin(), text.end(), text_sorted.begin());

    thrust::sort(text_sorted.begin(), text_sorted.end());
    thrust::device_vector<unsigned char> in_vals(256);
    thrust::device_vector<unsigned char> out_keys(256);
    thrust::device_vector<unsigned int> out_counts(256);

    thrust::fill(in_vals.begin(), in_vals.end(), 1);
    thrust::fill(out_counts.begin(), out_counts.end(), 0);

    auto new_end = thrust::reduce_by_key(text.begin(), text.end(),
      in_vals.begin(), out_keys.begin(), out_counts.begin());

    auto first = thrust::make_zip_iterator(
      thrust::make_tuple(out_counts.begin(), out_keys.begin())
    );

    auto last = thrust::make_zip_iterator(
      thrust::make_tuple(new_end.second, new_end.first)
    );

    return freq_alpha_lower;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Didn't supply plain text and period!" << std::endl;
        return 1;
    }

    // open file and verify
    std::ifstream ifs(argv[1], std::ios::binary);
    if (!ifs.good())
    {
        std::cerr << "Couldn't open book file!" << std::endl;
        return 1;
    }

    unsigned int period = std::atoi(argv[2]);
    if (period < 4)
    {
        std::cerr << "Period must be at least 4!" << std::endl;
        return 1;
    }

    std::vector<unsigned char> text;

    ifs.seekg(0, std::ios::end); // seek to end of file
    int length = ifs.tellg();    // get distance from beginning
    ifs.seekg(0, std::ios::beg); // move back to beginning

    text.resize(length);
    ifs.read((char *) &text[0], length);

    ifs.close();

    thrust::device_vector<unsigned char> text_clean;
    // TODO: sanitize input to contain only a-z lowercase (use the
    // isnot_lowercase_alpha functor), calculate the number of characters
    // in the cleaned text and put the result in text_clean, make sure to
    // resize text_clean to the correct size!
    int numElements = -1;

    std::cout << std::endl << "Before ciphering!" << std::endl << std::endl;
    std::vector<double> letterFreqGpu = getLetterFrequencyGpu(text_clean);
    std::vector<double> letterFreqCpu = getLetterFrequencyCpu(text);
    bool success = true;
    EXPECT_VECTOR_EQ_EPS(letterFreqCpu, letterFreqGpu, 1e-14, &success);
    PRINT_SUCCESS(success);

    thrust::device_vector<unsigned int> shifts(period);
    // TODO fill in shifts using thrust random number generation (make sure
    // not to allow 0-shifts, this would make for rather poor encryption).

    std::cout << std::endl << "Encryption key: ";

    for (unsigned int i = 0; i < period; ++i)
        std::cout << static_cast<char>('a' + shifts[i]);
    std::cout << std::endl;

    thrust::device_vector<unsigned char> device_cipher_text(numElements);

    // TODO: Apply the shifts to text_clean and place the result in
    // device_cipher_text.

    thrust::host_vector<unsigned char> host_cipher_text = device_cipher_text;

    std::cout << std::endl << "After ciphering!" << std::endl << std::endl;
    getLetterFrequencyGpu(device_cipher_text);

    std::ofstream ofs("cipher_text.txt", std::ios::binary);
    ofs.write((char *) &host_cipher_text[0], numElements);

    ofs.close();
    return 0;
}
