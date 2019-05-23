CUD=nvcc
CUDFLAGS=-O3 -arch=sm_30 -Xcompiler -Wall,-Wextra,-Wno-strict-aliasing -std=c++11
CIPHER_PERIOD=11

all: create_cipher solve_cipher

create_cipher: create_cipher.cu
	$(CUD) -o $@ $< $(CUDFLAGS)

solve_cipher: solve_cipher.cu
	$(CUD) -o $@ $< $(CUDFLAGS)

test: create_cipher solve_cipher
	./create_cipher mobydick.txt $(CIPHER_PERIOD)
	./solve_cipher cipher_text.txt

clean:
	rm -f *.o *~ *~ create_cipher solve_cipher
	rm -rf *.dSYM
