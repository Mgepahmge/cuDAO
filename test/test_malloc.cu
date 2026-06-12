#include <gtest/gtest.h>

#include <cuDAO.cuh>
#include <cuda.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace {

__global__ void fillSequenceKernel(int* data, int base, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        data[idx] = base + idx;
    }
}

__global__ void addValueKernel(int* data, int value, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        data[idx] += value;
    }
}

class PrimaryContextGuard {
public:
    bool init() noexcept {
        if (cuInit(0) != CUDA_SUCCESS) {
            return false;
        }

        if (cuDeviceGet(&device_, 0) != CUDA_SUCCESS) {
            return false;
        }

        if (cuDevicePrimaryCtxRetain(&ctx_, device_) != CUDA_SUCCESS) {
            return false;
        }

        retained_ = true;

        if (cuCtxSetCurrent(ctx_) != CUDA_SUCCESS) {
            cuDevicePrimaryCtxRelease(device_);
            retained_ = false;
            return false;
        }

        return true;
    }

    ~PrimaryContextGuard() {
        if (retained_) {
            cuDevicePrimaryCtxRelease(device_);
        }
    }

    CUdevice device() const noexcept {
        return device_;
    }

private:
    CUdevice device_{};
    CUcontext ctx_{};
    bool retained_{false};
};

std::vector<int> makeSequence(int n, int base) {
    std::vector<int> values(static_cast<size_t>(n));
    std::iota(values.begin(), values.end(), base);
    return values;
}

std::vector<int> snapshot(const int* data, int n) {
    return std::vector<int>(data, data + n);
}

void fillSequence(int* data, int n, int base) {
    for (int i = 0; i < n; ++i) {
        data[i] = base + i;
    }
}

void expectMemoryType(void* ptr, CUmemorytype expectedType) {
    ASSERT_NE(ptr, nullptr);

    CUmemorytype actualType{};
    ASSERT_EQ(cuPointerGetAttribute(
                  &actualType,
                  CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                  reinterpret_cast<CUdeviceptr>(ptr)
              ),
              CUDA_SUCCESS);
    EXPECT_EQ(actualType, expectedType);
}

void expectDeviceValues(int* devicePtr, const std::vector<int>& expected) {
    std::vector<int> actual(expected.size(), 0);
    ASSERT_EQ(cuMemcpyDtoH(
                  actual.data(),
                  reinterpret_cast<CUdeviceptr>(devicePtr),
                  sizeof(int) * actual.size()
              ),
              CUDA_SUCCESS);
    EXPECT_EQ(actual, expected);
}

void expectManagedMemorySupportedOrSkip(CUdevice device) {
    int managedMemorySupported = 0;
    ASSERT_EQ(cuDeviceGetAttribute(
                  &managedMemorySupported,
                  CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
                  device
              ),
              CUDA_SUCCESS);

    if (!managedMemorySupported) {
        GTEST_SKIP() << "Managed memory is not supported by this CUDA device";
    }
}

void expectMemoryPoolsSupportedOrSkip(CUdevice device) {
    int memoryPoolsSupported = 0;
    ASSERT_EQ(cuDeviceGetAttribute(
                  &memoryPoolsSupported,
                  CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                  device
              ),
              CUDA_SUCCESS);

    if (!memoryPoolsSupported) {
        GTEST_SKIP() << "Stream-ordered memory allocation is not supported by this CUDA device";
    }
}

#define ASSERT_CUDAO_SUCCESS(expr)                                           \
    do {                                                                    \
        const auto status = (expr);                                         \
        ASSERT_EQ(status.err, cuDAO::cuDAOError::Success)                   \
            << "cuDAOStatus error code: " << static_cast<int>(status.err)   \
            << ", CUDA result: " << static_cast<int>(status.cudaResult);    \
    } while (false)

}  // namespace

TEST(cuDAO, CuDAOMallocDeviceRegistersPointerForSchedulerUse) {
    PrimaryContextGuard context;
    if (!context.init()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    constexpr int n = 256;
    constexpr size_t bytes = sizeof(int) * n;

    int* devicePtr = nullptr;

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMalloc(
        &devicePtr,
        bytes,
        cuDAO::cuDAOMemKind::Device
    ));

    expectMemoryType(devicePtr, CU_MEMORYTYPE_DEVICE);

    const dim3 block{128, 1, 1};
    const dim3 grid{
        static_cast<unsigned int>((n + static_cast<int>(block.x) - 1) /
                                  static_cast<int>(block.x)),
        1,
        1
    };

    ASSERT_CUDAO_SUCCESS(cuDAO::launchKernel(
        fillSequenceKernel,
        grid,
        block,
        0,
        cuDAO::write(devicePtr),
        1000,
        n
    ));

    ASSERT_CUDAO_SUCCESS(cuDAO::sync(devicePtr));

    expectDeviceValues(devicePtr, makeSequence(n, 1000));

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOfree(devicePtr));
    ASSERT_CUDAO_SUCCESS(cuDAO::deviceSynchronize());
}

TEST(cuDAO, CuDAOMallocSupportsHostAndUnifiedMemoryKinds) {
    PrimaryContextGuard context;
    if (!context.init()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    constexpr int n = 256;
    constexpr size_t bytes = sizeof(int) * n;

    int* hostPtr = nullptr;
    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMalloc(
        &hostPtr,
        bytes,
        cuDAO::cuDAOMemKind::Host
    ));

    expectMemoryType(hostPtr, CU_MEMORYTYPE_HOST);

    fillSequence(hostPtr, n, 2000);
    EXPECT_EQ(snapshot(hostPtr, n), makeSequence(n, 2000));

    // Let the Register task observe the pointer value before releasing the
    // pinned host allocation. Do not call sync(hostPtr): host pointers are not
    // dependency-tracked by design.
    ASSERT_CUDAO_SUCCESS(cuDAO::deviceSynchronize());
    ASSERT_EQ(cuMemFreeHost(hostPtr), CUDA_SUCCESS);

    expectManagedMemorySupportedOrSkip(context.device());

    int* unifiedPtr = nullptr;
    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMalloc(
        &unifiedPtr,
        bytes,
        cuDAO::cuDAOMemKind::Unified
    ));

    expectMemoryType(unifiedPtr, CU_MEMORYTYPE_UNIFIED);

    const dim3 block{128, 1, 1};
    const dim3 grid{
        static_cast<unsigned int>((n + static_cast<int>(block.x) - 1) /
                                  static_cast<int>(block.x)),
        1,
        1
    };

    ASSERT_CUDAO_SUCCESS(cuDAO::launchKernel(
        fillSequenceKernel,
        grid,
        block,
        0,
        cuDAO::write(unifiedPtr),
        3000,
        n
    ));

    ASSERT_CUDAO_SUCCESS(cuDAO::sync(unifiedPtr));

    EXPECT_EQ(snapshot(unifiedPtr, n), makeSequence(n, 3000));

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOfree(unifiedPtr));
    ASSERT_CUDAO_SUCCESS(cuDAO::deviceSynchronize());
}

TEST(cuDAO, CuDAOMallocAsyncReturnsValidVirtualAddressBeforeSync) {
    PrimaryContextGuard context;
    if (!context.init()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    expectMemoryPoolsSupportedOrSkip(context.device());

    constexpr int n = 256;
    constexpr size_t bytes = sizeof(int) * n;

    int* devicePtr = nullptr;

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMallocAsync(&devicePtr, bytes));

    // cuDAOMallocAsync waits until cuMemAllocAsync has been invoked and the
    // pointer has received a valid virtual address. It does not, by itself,
    // imply that the stream-ordered allocation has completed.
    expectMemoryType(devicePtr, CU_MEMORYTYPE_DEVICE);

    // Explicitly wait for the allocation slot before using the pointer through
    // direct CUDA Driver API operations.
    ASSERT_CUDAO_SUCCESS(cuDAO::sync(devicePtr));

    ASSERT_EQ(cuMemsetD32(
                  reinterpret_cast<CUdeviceptr>(devicePtr),
                  0,
                  n
              ),
              CUDA_SUCCESS);

    expectDeviceValues(devicePtr, std::vector<int>(static_cast<size_t>(n), 0));

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOfree(devicePtr));
    ASSERT_CUDAO_SUCCESS(cuDAO::deviceSynchronize());
}

TEST(cuDAO, CuDAOMallocAsyncAllocationParticipatesInSchedulerDependencies) {
    PrimaryContextGuard context;
    if (!context.init()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    expectMemoryPoolsSupportedOrSkip(context.device());

    constexpr int n = 256;
    constexpr size_t bytes = sizeof(int) * n;

    int* devicePtr = nullptr;

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMallocAsync(&devicePtr, bytes));

    expectMemoryType(devicePtr, CU_MEMORYTYPE_DEVICE);

    const dim3 block{128, 1, 1};
    const dim3 grid{
        static_cast<unsigned int>((n + static_cast<int>(block.x) - 1) /
                                  static_cast<int>(block.x)),
        1,
        1
    };

    ASSERT_CUDAO_SUCCESS(cuDAO::launchKernel(
        fillSequenceKernel,
        grid,
        block,
        0,
        cuDAO::write(devicePtr),
        4000,
        n
    ));

    ASSERT_CUDAO_SUCCESS(cuDAO::launchKernel(
        addValueKernel,
        grid,
        block,
        0,
        cuDAO::write(devicePtr),
        17,
        n
    ));

    ASSERT_CUDAO_SUCCESS(cuDAO::sync(devicePtr));

    expectDeviceValues(devicePtr, makeSequence(n, 4017));

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOfree(devicePtr));
    ASSERT_CUDAO_SUCCESS(cuDAO::deviceSynchronize());
}
