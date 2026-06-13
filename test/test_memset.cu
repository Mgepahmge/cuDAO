#include <gtest/gtest.h>

#include <cuDAO.cuh>
#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {

__global__ void fillSequenceKernel(uint32_t* data, uint32_t base, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        data[idx] = base + static_cast<uint32_t>(idx);
    }
}

__global__ void addValueKernel(uint32_t* data, uint32_t value, int n) {
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

template <typename T>
std::vector<T> copyDeviceToHost(const T* devicePtr, size_t count) {
    std::vector<T> host(count);
    EXPECT_EQ(cuMemcpyDtoH(
                  host.data(),
                  reinterpret_cast<CUdeviceptr>(devicePtr),
                  sizeof(T) * count
              ),
              CUDA_SUCCESS);
    return host;
}

template <typename T>
std::vector<T> filledVector(size_t count, T value) {
    return std::vector<T>(count, value);
}

#define ASSERT_CUDAO_SUCCESS(expr)                                           \
    do {                                                                    \
        const auto status = (expr);                                         \
        ASSERT_EQ(status.err, cuDAO::cuDAOError::Success)                   \
            << "cuDAOStatus error code: " << static_cast<int>(status.err)   \
            << ", CUDA result: " << static_cast<int>(status.cudaResult);    \
    } while (false)

}  // namespace

TEST(cuDAO, CuDAOMemsetSupportsOneTwoAndFourByteValues) {
    PrimaryContextGuard context;
    if (!context.init()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    constexpr size_t n = 256;

    uint8_t* device8 = nullptr;
    uint16_t* device16 = nullptr;
    uint32_t* device32 = nullptr;

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMalloc(
        &device8,
        sizeof(uint8_t) * n,
        cuDAO::cuDAOMemKind::Device
    ));
    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMalloc(
        &device16,
        sizeof(uint16_t) * n,
        cuDAO::cuDAOMemKind::Device
    ));
    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMalloc(
        &device32,
        sizeof(uint32_t) * n,
        cuDAO::cuDAOMemKind::Device
    ));

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMemset(
        device8,
        static_cast<uint8_t>(0x5a),
        n
    ));
    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMemset(
        device16,
        static_cast<uint16_t>(0x1234),
        n
    ));
    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMemset(
        device32,
        static_cast<uint32_t>(0x89abcdefu),
        n
    ));

    ASSERT_CUDAO_SUCCESS(cuDAO::sync(device8));
    ASSERT_CUDAO_SUCCESS(cuDAO::sync(device16));
    ASSERT_CUDAO_SUCCESS(cuDAO::sync(device32));

    EXPECT_EQ(copyDeviceToHost(device8, n),
              filledVector<uint8_t>(n, static_cast<uint8_t>(0x5a)));
    EXPECT_EQ(copyDeviceToHost(device16, n),
              filledVector<uint16_t>(n, static_cast<uint16_t>(0x1234)));
    EXPECT_EQ(copyDeviceToHost(device32, n),
              filledVector<uint32_t>(n, static_cast<uint32_t>(0x89abcdefu)));

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOfree(device8));
    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOfree(device16));
    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOfree(device32));
    ASSERT_CUDAO_SUCCESS(cuDAO::deviceSynchronize());
}

TEST(cuDAO, CuDAOMemsetLazilyRegistersExternalDevicePointer) {
    PrimaryContextGuard context;
    if (!context.init()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    constexpr size_t n = 256;

    CUdeviceptr rawPtr{};
    ASSERT_EQ(cuMemAlloc(&rawPtr, sizeof(uint32_t) * n), CUDA_SUCCESS);

    auto* devicePtr = reinterpret_cast<uint32_t*>(rawPtr);

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMemset(
        devicePtr,
        static_cast<uint32_t>(0x01020304u),
        n
    ));

    ASSERT_CUDAO_SUCCESS(cuDAO::sync(devicePtr));

    EXPECT_EQ(copyDeviceToHost(devicePtr, n),
              filledVector<uint32_t>(n, static_cast<uint32_t>(0x01020304u)));

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOfree(devicePtr));
    ASSERT_CUDAO_SUCCESS(cuDAO::deviceSynchronize());
}

TEST(cuDAO, CuDAOMemsetParticipatesInSchedulerDependencies) {
    PrimaryContextGuard context;
    if (!context.init()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    constexpr int n = 256;
    constexpr size_t count = static_cast<size_t>(n);

    uint32_t* devicePtr = nullptr;

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMalloc(
        &devicePtr,
        sizeof(uint32_t) * count,
        cuDAO::cuDAOMemKind::Device
    ));

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
        static_cast<uint32_t>(1000),
        n
    ));

    // This write must wait for the previous kernel write and overwrite it.
    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMemset(
        devicePtr,
        static_cast<uint32_t>(42),
        count
    ));

    // This kernel must wait for memset to publish the next write version.
    ASSERT_CUDAO_SUCCESS(cuDAO::launchKernel(
        addValueKernel,
        grid,
        block,
        0,
        cuDAO::write(devicePtr),
        static_cast<uint32_t>(7),
        n
    ));

    ASSERT_CUDAO_SUCCESS(cuDAO::sync(devicePtr));

    EXPECT_EQ(copyDeviceToHost(devicePtr, count),
              filledVector<uint32_t>(count, static_cast<uint32_t>(49)));

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOfree(devicePtr));
    ASSERT_CUDAO_SUCCESS(cuDAO::deviceSynchronize());
}

TEST(cuDAO, CuDAOMemsetSupportsManagedMemoryTarget) {
    PrimaryContextGuard context;
    if (!context.init()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    expectManagedMemorySupportedOrSkip(context.device());

    constexpr size_t n = 256;

    uint32_t* managedPtr = nullptr;

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMalloc(
        &managedPtr,
        sizeof(uint32_t) * n,
        cuDAO::cuDAOMemKind::Unified
    ));

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMemset(
        managedPtr,
        static_cast<uint32_t>(0xaabbccddu),
        n
    ));

    ASSERT_CUDAO_SUCCESS(cuDAO::sync(managedPtr));

    EXPECT_EQ(std::vector<uint32_t>(managedPtr, managedPtr + n),
              filledVector<uint32_t>(n, static_cast<uint32_t>(0xaabbccddu)));

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOfree(managedPtr));
    ASSERT_CUDAO_SUCCESS(cuDAO::deviceSynchronize());
}
