#include <gtest/gtest.h>

#include <cuDAO.cuh>
#include <cuda.h>

#include <algorithm>
#include <numeric>
#include <variant>
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

class DeviceBuffer {
public:
    bool allocate(size_t bytes) noexcept {
        return cuMemAlloc(&ptr_, bytes) == CUDA_SUCCESS;
    }

    ~DeviceBuffer() {
        if (ptr_ != 0) {
            cuMemFree(ptr_);
        }
    }

    CUdeviceptr ptr() const noexcept {
        return ptr_;
    }

    template <typename T>
    T* as() const noexcept {
        return reinterpret_cast<T*>(ptr_);
    }

private:
    CUdeviceptr ptr_{};
};

class ManagedBuffer {
public:
    bool allocate(size_t bytes) noexcept {
        return cuMemAllocManaged(&ptr_, bytes, CU_MEM_ATTACH_GLOBAL) == CUDA_SUCCESS;
    }

    ~ManagedBuffer() {
        if (ptr_ != 0) {
            cuMemFree(ptr_);
        }
    }

    template <typename T>
    T* as() const noexcept {
        return reinterpret_cast<T*>(ptr_);
    }

private:
    CUdeviceptr ptr_{};
};

class PinnedHostBuffer {
public:
    bool allocate(size_t bytes) noexcept {
        void* raw = nullptr;
        if (cuMemAllocHost(&raw, bytes) != CUDA_SUCCESS) {
            return false;
        }

        ptr_ = raw;
        return true;
    }

    ~PinnedHostBuffer() {
        if (ptr_ != nullptr) {
            cuMemFreeHost(ptr_);
        }
    }

    template <typename T>
    T* as() const noexcept {
        return reinterpret_cast<T*>(ptr_);
    }

private:
    void* ptr_{nullptr};
};

std::vector<int> makeSequence(int n, int base) {
    std::vector<int> values(static_cast<size_t>(n));
    std::iota(values.begin(), values.end(), base);
    return values;
}

void fillSequence(int* data, int n, int base) {
    for (int i = 0; i < n; ++i) {
        data[i] = base + i;
    }
}

std::vector<int> snapshot(const int* data, int n) {
    return std::vector<int>(data, data + n);
}

cuDAO::CudaFuture expectFuture(
    std::variant<cuDAO::cuDAOStatus, cuDAO::CudaFuture>& result
) {
    if (const auto* status = std::get_if<cuDAO::cuDAOStatus>(&result)) {
        ADD_FAILURE() << "Expected CudaFuture, got cuDAOStatus error code: "
                      << static_cast<int>(status->err);
        return cuDAO::CudaFuture{std::make_shared<cuDAO::CudaPromise>()};
    }

    return std::get<cuDAO::CudaFuture>(result);
}

#define ASSERT_CUDAO_SUCCESS(expr)                                           \
    do {                                                                    \
        const auto status = (expr);                                         \
        ASSERT_EQ(status.err, cuDAO::cuDAOError::Success)                   \
            << "cuDAOStatus error code: " << static_cast<int>(status.err);  \
    } while (false)

#define ASSERT_CUDAO_FUTURE(expr)                                           \
    do {                                                                    \
        auto result = (expr);                                               \
        ASSERT_TRUE(std::holds_alternative<cuDAO::CudaFuture>(result));     \
        auto future = std::get<cuDAO::CudaFuture>(result);                  \
        future.wait();                                                      \
    } while (false)

}  // namespace

TEST(cuDAO, CuDAOMemcpyExplicitDirectionsRoundTrip) {
    PrimaryContextGuard context;
    if (!context.init()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    constexpr int n = 256;
    constexpr size_t bytes = sizeof(int) * n;

    PinnedHostBuffer hostSrc;
    PinnedHostBuffer hostDst;
    DeviceBuffer deviceA;
    DeviceBuffer deviceB;

    ASSERT_TRUE(hostSrc.allocate(bytes));
    ASSERT_TRUE(hostDst.allocate(bytes));
    ASSERT_TRUE(deviceA.allocate(bytes));
    ASSERT_TRUE(deviceB.allocate(bytes));

    fillSequence(hostSrc.as<int>(), n, 100);
    std::fill(hostDst.as<int>(), hostDst.as<int>() + n, 0);

    ASSERT_EQ(cuMemsetD32(deviceA.ptr(), 0, n), CUDA_SUCCESS);
    ASSERT_EQ(cuMemsetD32(deviceB.ptr(), 0, n), CUDA_SUCCESS);

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMemcpy(
        deviceA.as<int>(),
        hostSrc.as<const int>(),
        bytes,
        cuDAO::cuDAOMemcpyType::HostToDevice
    ));

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMemcpy(
        deviceB.as<int>(),
        deviceA.as<const int>(),
        bytes,
        cuDAO::cuDAOMemcpyType::DeviceToDevice
    ));

    // Device-to-host writes to an untracked host pointer, so completion is
    // observed through the memcpy future instead of sync(hostDst).
    ASSERT_CUDAO_FUTURE(cuDAO::cuDAOMemcpySync(
        hostDst.as<int>(),
        deviceB.as<const int>(),
        bytes,
        cuDAO::cuDAOMemcpyType::DeviceToHost
    ));

    EXPECT_EQ(snapshot(hostDst.as<const int>(), n),
              snapshot(hostSrc.as<const int>(), n));
}

TEST(cuDAO, CuDAOMemcpyAutoDetectsHostDeviceDirections) {
    PrimaryContextGuard context;
    if (!context.init()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    constexpr int n = 256;
    constexpr size_t bytes = sizeof(int) * n;

    PinnedHostBuffer hostSrc;
    PinnedHostBuffer hostDst;
    DeviceBuffer deviceA;
    DeviceBuffer deviceB;

    ASSERT_TRUE(hostSrc.allocate(bytes));
    ASSERT_TRUE(hostDst.allocate(bytes));
    ASSERT_TRUE(deviceA.allocate(bytes));
    ASSERT_TRUE(deviceB.allocate(bytes));

    fillSequence(hostSrc.as<int>(), n, 200);
    std::fill(hostDst.as<int>(), hostDst.as<int>() + n, 0);

    ASSERT_EQ(cuMemsetD32(deviceA.ptr(), 0, n), CUDA_SUCCESS);
    ASSERT_EQ(cuMemsetD32(deviceB.ptr(), 0, n), CUDA_SUCCESS);

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMemcpy(
        deviceA.as<int>(),
        hostSrc.as<const int>(),
        bytes
    ));

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMemcpy(
        deviceB.as<int>(),
        deviceA.as<const int>(),
        bytes
    ));

    // Auto-detected device-to-host also completes through the returned future.
    ASSERT_CUDAO_FUTURE(cuDAO::cuDAOMemcpySync(
        hostDst.as<int>(),
        deviceB.as<const int>(),
        bytes
    ));

    EXPECT_EQ(snapshot(hostDst.as<const int>(), n),
              snapshot(hostSrc.as<const int>(), n));
}

TEST(cuDAO, CuDAOMemcpyAutoDetectsUnifiedMemoryDirections) {
    PrimaryContextGuard context;
    if (!context.init()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    int managedMemorySupported = 0;
    ASSERT_EQ(cuDeviceGetAttribute(
                  &managedMemorySupported,
                  CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
                  context.device()
              ),
              CUDA_SUCCESS);

    if (!managedMemorySupported) {
        GTEST_SKIP() << "Managed memory is not supported by this CUDA device";
    }

    constexpr int n = 256;
    constexpr size_t bytes = sizeof(int) * n;

    PinnedHostBuffer hostSrc;
    PinnedHostBuffer hostDst;
    DeviceBuffer deviceBuffer;
    ManagedBuffer managedSrc;
    ManagedBuffer managedDst;

    ASSERT_TRUE(hostSrc.allocate(bytes));
    ASSERT_TRUE(hostDst.allocate(bytes));
    ASSERT_TRUE(deviceBuffer.allocate(bytes));
    ASSERT_TRUE(managedSrc.allocate(bytes));
    ASSERT_TRUE(managedDst.allocate(bytes));

    fillSequence(hostSrc.as<int>(), n, 300);
    fillSequence(managedSrc.as<int>(), n, 400);
    std::fill(hostDst.as<int>(), hostDst.as<int>() + n, 0);
    std::fill(managedDst.as<int>(), managedDst.as<int>() + n, 0);

    ASSERT_EQ(cuMemsetD32(deviceBuffer.ptr(), 0, n), CUDA_SUCCESS);

    // Host -> Unified: destination is tracked, so sync(managedDst) is valid.
    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMemcpy(
        managedDst.as<int>(),
        hostSrc.as<const int>(),
        bytes
    ));
    ASSERT_CUDAO_SUCCESS(cuDAO::sync(managedDst.as<int>()));
    EXPECT_EQ(snapshot(managedDst.as<const int>(), n),
              snapshot(hostSrc.as<const int>(), n));

    // Unified -> Host: destination is untracked, so use the future.
    std::fill(hostDst.as<int>(), hostDst.as<int>() + n, 0);
    ASSERT_CUDAO_FUTURE(cuDAO::cuDAOMemcpySync(
        hostDst.as<int>(),
        managedSrc.as<const int>(),
        bytes
    ));
    EXPECT_EQ(snapshot(hostDst.as<const int>(), n),
              snapshot(managedSrc.as<const int>(), n));

    // Unified -> Unified: destination is tracked.
    std::fill(managedDst.as<int>(), managedDst.as<int>() + n, 0);
    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMemcpy(
        managedDst.as<int>(),
        managedSrc.as<const int>(),
        bytes
    ));
    ASSERT_CUDAO_SUCCESS(cuDAO::sync(managedDst.as<int>()));
    EXPECT_EQ(snapshot(managedDst.as<const int>(), n),
              snapshot(managedSrc.as<const int>(), n));

    // Unified -> Device: destination is tracked.
    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMemcpy(
        deviceBuffer.as<int>(),
        managedSrc.as<const int>(),
        bytes
    ));
    ASSERT_CUDAO_SUCCESS(cuDAO::sync(deviceBuffer.as<int>()));

    // Device -> Unified: destination is tracked.
    std::fill(managedDst.as<int>(), managedDst.as<int>() + n, 0);
    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMemcpy(
        managedDst.as<int>(),
        deviceBuffer.as<const int>(),
        bytes
    ));
    ASSERT_CUDAO_SUCCESS(cuDAO::sync(managedDst.as<int>()));

    EXPECT_EQ(snapshot(managedDst.as<const int>(), n),
              snapshot(managedSrc.as<const int>(), n));
}

TEST(cuDAO, CuDAOMemcpyParticipatesInSchedulerDependencies) {
    PrimaryContextGuard context;
    if (!context.init()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    constexpr int n = 256;
    constexpr size_t bytes = sizeof(int) * n;

    PinnedHostBuffer hostDst;
    DeviceBuffer deviceA;
    DeviceBuffer deviceB;

    ASSERT_TRUE(hostDst.allocate(bytes));
    ASSERT_TRUE(deviceA.allocate(bytes));
    ASSERT_TRUE(deviceB.allocate(bytes));

    std::fill(hostDst.as<int>(), hostDst.as<int>() + n, 0);

    ASSERT_EQ(cuMemsetD32(deviceA.ptr(), 0, n), CUDA_SUCCESS);
    ASSERT_EQ(cuMemsetD32(deviceB.ptr(), 0, n), CUDA_SUCCESS);

    const dim3 block{128, 1, 1};
    const dim3 grid{
        static_cast<unsigned int>((n + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
        1,
        1
    };

    ASSERT_CUDAO_SUCCESS(cuDAO::launchKernel(
        fillSequenceKernel,
        grid,
        block,
        0,
        cuDAO::write(deviceA.as<int>()),
        500,
        n
    ));

    ASSERT_CUDAO_SUCCESS(cuDAO::cuDAOMemcpy(
        deviceB.as<int>(),
        deviceA.as<const int>(),
        bytes,
        cuDAO::cuDAOMemcpyType::DeviceToDevice
    ));

    ASSERT_CUDAO_SUCCESS(cuDAO::launchKernel(
        addValueKernel,
        grid,
        block,
        0,
        cuDAO::write(deviceB.as<int>()),
        7,
        n
    ));

    // Final host readback is not synchronized with sync(hostDst).
    ASSERT_CUDAO_FUTURE(cuDAO::cuDAOMemcpySync(
        hostDst.as<int>(),
        deviceB.as<const int>(),
        bytes,
        cuDAO::cuDAOMemcpyType::DeviceToHost
    ));

    EXPECT_EQ(snapshot(hostDst.as<const int>(), n), makeSequence(n, 507));
}