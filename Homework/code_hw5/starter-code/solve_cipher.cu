#include <vector>
#include <fstream>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/extrema.h>

#include "strided_range_iterator.h"

// You will need to call these functors from thrust functions in the code
// do not create new ones

// this can be the same as in create_cipher.cu
// apply a shift with appropriate wrapping
struct apply_shift : thrust::binary_function<unsigned char, int, unsigned char>
{
    thrust::device_ptr<int> shifts;
    unsigned int period;

  public:
    apply_shift(thrust::device_ptr<int> shifts_, unsigned int period_){
      period = period_;
      shifts = shifts_;
    }

    __host__ __device__
    unsigned char operator()(const unsigned char& x, const int& loc){
      unsigned char y = x + shifts[loc%period];
      int y_int = int(y);
      if (y_int <= 97){
         y_int = 97 + 26 - (97-y_int)%26;
      }
      if (y_int > 97){
         y_int = 97 + (y_int-97)%26;
      }
      return static_cast<unsigned char>(y_int);
    }
};


struct matchElems : thrust::binary_function<unsigned char, unsigned char, int>
{
    __host__ __device__
    int operator()(const unsigned char& x, const unsigned char& y){
      if (x==y){
        return 1;
      }
      return 0;
    }
};


int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "No cipher text given!" << std::endl;
        return 1;
    }

    // First load the text
    std::ifstream ifs(argv[1], std::ios::binary);

    if (!ifs.good())
    {
        std::cerr << "Couldn't open book file!" << std::endl;
        return 1;
    }

    std::vector<unsigned char> text;

    ifs.seekg(0, std::ios::end); // seek to end of file
    int length = ifs.tellg();    // get distance from beginning
    ifs.seekg(0, std::ios::beg); // move back to beginning

    text.resize(length);
    ifs.read((char *) &text[0], length);

    ifs.close();

    // we assume the cipher text has been sanitized
    thrust::device_vector<unsigned char> text_clean = text;

    // now we crack the Vigenere cipher
    // first we need to determine the key length
    // use the kappa index of coincidence
    int keyLength = 0;
    {
        bool found = false;
        int shift_idx = 4; // Start at index 4.

        while (!found)
        //for(int i = 0; i < 10; i++)
        {
            // TODO: Use thrust to compute the number of characters that match
            // when shifting text_clean by shift_idx.
            thrust::device_vector<unsigned char> text_shift(text_clean.size()-shift_idx);
            thrust::device_vector<int> matches(text_clean.size()-shift_idx);
            thrust::copy(text_clean.begin()+shift_idx, text_clean.end(),text_shift.begin());

            thrust::transform(text_shift.begin(), text_shift.end(), text_clean.begin(),
              matches.begin(), matchElems());

            int numMatches = thrust::reduce(matches.begin(), matches.end());
            std::cout << "shift " << shift_idx << ", num matches " << numMatches << "\n";
            double ioc = numMatches / (static_cast<double>(text_clean.size() - shift_idx) / 26.);

            std::cout << "Period " << shift_idx << " ioc: " << ioc << std::endl;

            if (ioc > 1.6)
            {
                if (keyLength == 0)
                {
                    keyLength = shift_idx;
                    shift_idx = 2 * shift_idx - 1; // check double the period to make sure
                }
                else if (2 * keyLength == shift_idx)
                {
                    found = true;
                }
                else
                {
                    std::cout << "Unusual pattern in text! Probably period is < 4."
                              << std::endl;
                    exit(1);
                }
            }

            ++shift_idx;
        }
    }

    std::cout << "keyLength: " << keyLength << std::endl;

    // once we know the key length, then we can do frequency analysis on each
    // pos mod length allowing us to easily break each cipher independently
    // you will find the strided_range useful
    // it is located in strided_range_iterator.h and an example
    // of how to use it is located in the that file
    thrust::device_vector<unsigned char> text_copy = text_clean;
    thrust::device_vector<int> dShifts(keyLength);
    using Iterator = typename thrust::device_vector<unsigned char>::iterator;


    // TODO: Now that you have determined the length of the key, you need to
    // compute the actual key. To do so, perform keyLength individual frequency
    // analyses on text_copy to find the shift which aligns the most common
    // character in text_copy with the character 'e'. Fill up the
    // dShifts vector with the correct shifts.
    std::cout << "text length " << text_copy.size() << "\n";
    for(int i = 0; i < keyLength; i++){
      auto it = strided_range<Iterator>(text_copy.begin()+i, text_copy.end(), keyLength);

      thrust::device_vector<unsigned char> strid_text(text_copy.size()/keyLength, 'a');
      thrust::copy(it.begin(), it.end(), strid_text.begin());

      thrust::sort(strid_text.begin(), strid_text.end());
      thrust::device_vector<unsigned char> out_keys(26);
      thrust::device_vector<unsigned int> out_counts(26);

      auto new_end = thrust::reduce_by_key(strid_text.begin(), strid_text.end(),
        thrust::make_constant_iterator(1), out_keys.begin(), out_counts.begin());

      out_keys.erase(new_end.first, out_keys.end());
      out_counts.erase(new_end.second, out_counts.end());

      thrust::transform(out_counts.begin(),out_counts.end(),
        out_counts.begin(), thrust::negate<int>());

      thrust::sort_by_key(out_counts.begin(), new_end.second, out_keys.begin());

      thrust::transform(out_counts.begin(),out_counts.end(),
        out_counts.begin(), thrust::negate<int>());

      dShifts[i] = -(out_keys[0]-'e');
    }


    std::cout << std::endl << "Encryption key: ";

    for (int i = 0; i < keyLength; ++i)
    {
        int keyval = 'a' - (dShifts[i] <= 0 ? dShifts[i] : dShifts[i] - 26);
        std::cout << static_cast<char>(keyval);
    }
    std::cout << std::endl;

    // take the shifts and transform cipher text back to plain text
    // TODO : transform the cipher text back to the plain text by using the
    // apply_shift functor.
    thrust::device_vector<int> seq(text_clean.size());
    thrust::sequence(seq.begin(),seq.end());

    auto apl = apply_shift(dShifts.data(), keyLength);
    thrust::transform(text_clean.begin(), text_clean.end(), seq.begin(),
      text_clean.begin(), apl);
    thrust::host_vector<unsigned char> h_plain_text = text_clean;

    std::ofstream ofs("plain_text.txt", std::ios::binary);
    ofs.write((char *) &h_plain_text[0], h_plain_text.size());
    ofs.close();
    return 0;
}
