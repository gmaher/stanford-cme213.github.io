#include <stdio.h>

// Print device properties
void PrintDevProp(cudaDeviceProp dev_prop) {
    printf("Major revision number:         %d\n",  dev_prop.major);
    printf("Minor revision number:         %d\n",  dev_prop.minor);
    printf("Name:                          %s\n",  dev_prop.name);
    printf("Total global memory:           %lu\n", dev_prop.totalGlobalMem);
    printf("Total shared memory per block: %lu\n", dev_prop.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  dev_prop.regsPerBlock);
    printf("Warp size:                     %d\n",  dev_prop.warpSize);
    printf("Maximum memory pitch:          %lu\n", dev_prop.memPitch);
    printf("Maximum threads per block:     %d\n",  dev_prop.maxThreadsPerBlock);

    for(int i = 0; i < 3; ++i) {
        printf("Maximum dimension %d of block:  %d\n", i, dev_prop.maxThreadsDim[i]);
    }

    for(int i = 0; i < 3; ++i) {
        printf("Maximum dimension %d of grid:   %d\n", i, dev_prop.maxGridSize[i]);
    }

    printf("Clock rate:                    %d\n",  dev_prop.clockRate);
    printf("Total constant memory:         %lu\n", dev_prop.totalConstMem);
    printf("Texture alignment:             %lu\n", dev_prop.textureAlignment);
    printf("Concurrent copy and execution: %s\n",
           (dev_prop.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  dev_prop.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",
           (dev_prop.kernelExecTimeoutEnabled ? "Yes" : "No"));
    printf("ECC enabled:      %s\n",
           (dev_prop.ECCEnabled ? "Yes" : "No"));
    return;
}

int main() {
    // Number of CUDA devices
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", dev_count);

    // Iterate through devices
    for(int i = 0; i < dev_count; ++i) {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, i);
        PrintDevProp(dev_prop);
    }

    return 0;
}
