#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random.h>
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
      if (x>='a' && x<='z'){return false;}
      return true;
    }
};

// convert an uppercase letter into a lowercase one
// do not use the builtin C function or anything from boost, etc.
struct upper_to_lower : thrust::unary_function<unsigned char, unsigned char>
{
    __host__ __device__
    unsigned char operator()(const unsigned char& x) const
    {
      if(x>='A' && x<='Z'){
        return x+32;
      }
      return x;
    }
};

// apply a shift with appropriate wrapping
struct apply_shift : thrust::binary_function<unsigned char, int, unsigned char>
{
    thrust::device_ptr<unsigned int> shifts;
    unsigned int period;

  public:
    apply_shift(thrust::device_ptr<unsigned int> shifts_, unsigned int period_){
      period = period_;
      shifts = shifts_;
    }

    __host__ __device__
    unsigned char operator()(const unsigned char& x, const int& loc){
      return x;
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
    thrust::device_vector<unsigned char> out_keys(26);
    thrust::device_vector<unsigned int> out_counts(26);

    auto new_end = thrust::reduce_by_key(text_sorted.begin(), text_sorted.end(),
      thrust::make_constant_iterator(1), out_keys.begin(), out_counts.begin());

    out_keys.erase(new_end.first, out_keys.end());
    out_counts.erase(new_end.second, out_counts.end());

    thrust::transform(out_counts.begin(),out_counts.end(),
      out_counts.begin(), thrust::negate<int>());

    thrust::sort_by_key(out_counts.begin(), new_end.second, out_keys.begin());

    thrust::transform(out_counts.begin(),out_counts.end(),
      out_counts.begin(), thrust::negate<int>());

    int sum = thrust::reduce(out_counts.begin(), out_counts.end());
    std::cout << "sum=" << sum << "\n";

    auto key_it = out_keys.begin();
    auto count_it = out_counts.begin();

    std::vector<unsigned char> h_keys;

    for (int i = 0; i <5; i++){
      h_keys.push_back(out_keys[i]);
      freq_alpha_lower.push_back(out_counts[i]*1.0/sum);

      printf("%c %f\n", h_keys[i], freq_alpha_lower[i]);
      key_it++;
      count_it++;
      if (key_it == new_end.first){
        break;
      }
    }

    return freq_alpha_lower;
}

int rand(void)
{
  static thrust::default_random_engine rng;
  static thrust::uniform_int_distribution<int> dist(1, 25);
  return dist(rng);
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

    int period = std::atoi(argv[2]);
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

    // TODO: sanitize input to contain only a-z lowercase (use the
    // isnot_lowercase_alpha functor), calculate the number of characters
    // in the cleaned text and put the result in text_clean, make sure to
    // resize text_clean to the correct size!
    thrust::device_vector<unsigned char> d_text(text);

    thrust::transform_if(d_text.begin(), d_text.end(), d_text.begin(),
      upper_to_lower(), isnot_lowercase_alpha());

    auto new_end = thrust::remove_if(d_text.begin(), d_text.end(), isnot_lowercase_alpha());

    thrust::device_vector<unsigned char> text_clean(d_text.begin(), new_end);

    int numElements = text_clean.size();

    std::cout << std::endl << "Before ciphering!" << std::endl << std::endl;
    std::vector<double> letterFreqGpu = getLetterFrequencyGpu(text_clean);
    std::vector<double> letterFreqCpu = getLetterFrequencyCpu(text);
    bool success = true;
    EXPECT_VECTOR_EQ_EPS(letterFreqCpu, letterFreqGpu, 1e-14, &success);
    PRINT_SUCCESS(success);

    thrust::device_vector<unsigned int> shifts(period);
    // TODO fill in shifts using thrust random number generation (make sure
    // not to allow 0-shifts, this would make for rather poor encryption).
    thrust::generate(shifts.begin(),shifts.end(), rand);


    std::cout << std::endl << "Encryption key: ";

    // for (unsigned int i = 0; i < period; ++i)
    //     std::cout << static_cast<char>('a' + shifts[i]);
    // std::cout << std::endl;

    thrust::device_vector<unsigned char> device_cipher_text(numElements);

    thrust::device_vector<int> seq(numElements);
    std::cout << "done shifting\n";
    thrust::sequence(seq.begin(),seq.end());

    // TODO: Apply the shifts to text_clean and place the result in
    // device_cipher_text.
    auto apl = apply_shift(shifts.data(), (unsigned int)period);
    std::cout << "apl " << apl('a',2) << "\n";

    std::cout << "done shifting\n";
    thrust::transform(text_clean.begin(), text_clean.end(), seq.begin(), text_clean.begin(),
      apl);
    std::cout << "done shifting\n";
    thrust::host_vector<unsigned char> host_cipher_text(numElements);
    std::cout << "a " << host_cipher_text[0] << "\n";
    std::cout << "a " << device_cipher_text.data()[0] << "\n";

    thrust::copy(device_cipher_text.begin(), device_cipher_text.end(),
      host_cipher_text.begin());

    std::cout << std::endl << "After ciphering!" << std::endl << std::endl;
    getLetterFrequencyGpu(device_cipher_text);

    std::ofstream ofs("cipher_text.txt", std::ios::binary);
    ofs.write((char *) &host_cipher_text[0], numElements);

    ofs.close();
    return 0;
}
