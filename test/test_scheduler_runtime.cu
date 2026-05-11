#include <gtest/gtest.h>
#include <cuDAO.cuh>
#include <cuda.h>

namespace {
    __global__ void writeValueKernel(int* data, int value) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            data[0] = value;
        }
    }

    bool initCudaOrSkip(CUdevice& device, CUcontext& ctx) {
        if (cuInit(0) != CUDA_SUCCESS) {
            return false;
        }
        if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
            return false;
        }
        if (cuDevicePrimaryCtxRetain(&ctx, device) != CUDA_SUCCESS) {
            return false;
        }
        if (cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
            cuDevicePrimaryCtxRelease(device);
            return false;
        }
        return true;
    }
}

TEST(cuDAO, SchedulerLaunchKernelSyncCompletesAndWrites) {
    CUdevice device{};
    CUcontext ctx{};
    if (!initCudaOrSkip(device, ctx)) {
        GTEST_SKIP();
    }

    CUdeviceptr devicePtr{};
    ASSERT_EQ(cuMemAlloc(&devicePtr, sizeof(int)), CUDA_SUCCESS);
    ASSERT_EQ(cuMemsetD32(devicePtr, 0, 1), CUDA_SUCCESS);

    auto future = cuDAO::launchKernelSync(
        writeValueKernel,
        dim3{1, 1, 1},
        dim3{1, 1, 1},
        0,
        cuDAO::write(reinterpret_cast<int*>(devicePtr)),
        42
    );

    EXPECT_FALSE(future.ready());
    future.wait();
    EXPECT_TRUE(future.ready());

    int hostValue = 0;
    ASSERT_EQ(cuMemcpyDtoH(&hostValue, devicePtr, sizeof(hostValue)), CUDA_SUCCESS);
    EXPECT_EQ(hostValue, 42);

    cuMemFree(devicePtr);
    cuDevicePrimaryCtxRelease(device);
}

TEST(cuDAO, SchedulerLaunchKernelCompletesAndWrites) {
    CUdevice device{};
    CUcontext ctx{};
    if (!initCudaOrSkip(device, ctx)) {
        GTEST_SKIP();
    }

    CUdeviceptr devicePtr{};
    ASSERT_EQ(cuMemAlloc(&devicePtr, sizeof(int)), CUDA_SUCCESS);
    ASSERT_EQ(cuMemsetD32(devicePtr, 0, 1), CUDA_SUCCESS);

    cuDAO::launchKernel(
        writeValueKernel,
        dim3{1, 1, 1},
        dim3{1, 1, 1},
        0,
        cuDAO::write(reinterpret_cast<int*>(devicePtr)),
        7
    );

    cuDAO::deviceSynchronize();

    int hostValue = 0;
    ASSERT_EQ(cuMemcpyDtoH(&hostValue, devicePtr, sizeof(hostValue)), CUDA_SUCCESS);
    EXPECT_EQ(hostValue, 7);

    cuMemFree(devicePtr);
    cuDevicePrimaryCtxRelease(device);
}
