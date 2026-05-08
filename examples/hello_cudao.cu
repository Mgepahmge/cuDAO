#include <cuDAO.cuh>
#include <cuda.h>
#include <cstdio>

int main() {
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed: %d\n", res);
        return 1;
    }

    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);
    printf("cuDAO v%d — environment OK, %d device(s) found.\n",
           CUDAO_VERSION, deviceCount);
    return 0;
}
