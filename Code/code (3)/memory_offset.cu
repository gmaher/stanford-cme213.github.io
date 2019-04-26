__global__ void offsetCopy(float* odata, float* idata, int offset) {
    int xid = blockIdx.x * blockDim.x + threadIdx.x + offset;
    odata[xid] = idata[xid];
}

__global__ void stridedCopy(float* odata, float* idata, int stride) {
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    odata[xid] = idata[stride*xid];
}

__global__ void randomCopy(float* odata, float* idata, int* addr) {
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    odata[xid] = idata[addr[xid]];
}

__global__ void tid(int * wID) {

int tID = threadIdx.x 
	+ threadIdx.y * blockDim.x 
    + threadIdx.z * blockDim.x * blockDim.y;
int warpID = tID/32;

*wID = warpID;
}

int main(void) {
    return 0;
}
