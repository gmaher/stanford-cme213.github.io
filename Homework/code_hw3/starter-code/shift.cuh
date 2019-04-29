#ifndef _SHIFT_CUH
#define _SHIFT_CUH

/**
 * Repeating from the tutorial, just in case you haven't looked at it.
 * "kernels" or __global__ functions are the entry points to code that executes on the GPU.
 * The keyword __global__ indicates to the compiler that this function is a GPU entry point.
 * __global__ functions must return void, and may only be called or "launched" from code that
 * executes on the CPU.
 */

typedef unsigned char uchar;
typedef unsigned int uint;

/**
 * Implements a per-element shift by loading a single byte and shifting it.
 */
__global__ void shift_char(const uchar *input_array, uchar *output_array,
                           uchar shift_amount, uint array_length)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < array_length){
    output_array[i] = input_array[i]+shift_amount;
  }
}

/**
 * Here we load 4 bytes at a time instead of just 1 to improve bandwidth
 * due to a better memory access pattern.
 */
__global__ void shift_int(const uint *input_array, uint *output_array,
                          uint shift_amount, uint array_length)
{
    uint i = blockIdx.x*blockDim.x + threadIdx.x;

    if (4*i < array_length){
      output_array[i] = input_array[i]+shift_amount;
    }
}

/**
 * Here we go even further and load 8 bytes - does it improve further?
 */
__global__ void shift_int2(const uint2 *input_array, uint2 *output_array,
                           uint shift_amount, uint array_length)
{
    uint i = blockIdx.x*blockDim.x + threadIdx.x;
    if (8*i < array_length){
      output_array[i].x = input_array[i].x+shift_amount;
      output_array[i].y = input_array[i].y+shift_amount;
    }
}

// the following three kernels launch their respective kernels
// and report the time it took for the kernel to run

double doGPUShiftChar(const uchar *d_input, uchar *d_output,
                      uchar shift_amount, uint text_size, uint block_size)
{
    uint arr_size = text_size*sizeof(unsigned char);

    uint num_blocks = arr_size/block_size+1;

    // TODO: compute 4 byte shift value
    event_pair timer;
    start_timer(&timer);

    // TODO: launch kernel
    shift_char<<<num_blocks, block_size>>>(d_input, d_output,
        shift_amount, arr_size);

    check_launch("gpu shift cipher uint");
    return stop_timer(&timer);
}

double doGPUShiftUInt(const uchar *d_input, uchar *d_output,
                      uchar shift_amount, uint text_size, uint block_size)
{
    // TODO: compute grid dimensions
    //since each uint is 4 uchars, we only need 1/4 the blocks
    uint num_blocks = (text_size/4)/block_size+1;

    // TODO: compute 4 byte shift value
    uint shift_amount_int = (uint)(
      (shift_amount << 24) | (shift_amount << 16) | (shift_amount << 8) |
      (shift_amount) );

    event_pair timer;
    start_timer(&timer);

    shift_int<<<num_blocks,block_size>>>((uint*)d_input, (uint*)d_output,
      shift_amount_int, text_size);

    check_launch("gpu shift cipher uint");
    return stop_timer(&timer);
}

double doGPUShiftUInt2(const uchar* d_input, uchar* d_output,
                       uchar shift_amount, uint text_size, uint block_size)
{
    // TODO: compute your grid dimensions
    uint num_blocks = (text_size/8)/block_size+1;

    // TODO: compute 4 byte shift value
    uint shift_amount_int = (uint)(
      (shift_amount << 24) | (shift_amount << 16) | (shift_amount << 8) |
      (shift_amount) );

    event_pair timer;
    start_timer(&timer);

    // TODO: launch kernel
    shift_int2<<<num_blocks,block_size>>>((uint2*)d_input, (uint2*)d_output,
      shift_amount_int, text_size);

    check_launch("gpu shift cipher uint2");
    return stop_timer(&timer);
}


#endif
