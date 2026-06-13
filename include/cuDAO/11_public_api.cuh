#pragma once
#include "10_Scheduler.cuh"

namespace cuDAO {

    // ──────────────────────────────────────────────────────────────────────────
    // Public API
    // ──────────────────────────────────────────────────────────────────────────
    inline cuDAOStatus cuDAOInit() noexcept {
        const auto& scheduler = getDefaultScheduler();
        return cuDAOStatus{scheduler.initStatus};
    }

    inline std::optional<cuDAOStatus> cuDAOGetLastError() noexcept {
        auto& errorQueue = getErrorQueue();
        cuDAOStatus status{};
        if (!errorQueue.pop(status)) {
            return std::nullopt;
        }
        return cuDAOStatus{status};
    }

    template <typename Func, typename... Args>
    cuDAOStatus launchKernel(Func func, dim3 grid, dim3 block, size_t sharedMem, Args&&... args) noexcept {
        try {
            auto task = buildTask(func, grid, block, sharedMem, std::forward<Args>(args)...);
            auto& scheduler = getDefaultScheduler();
            if (scheduler.initStatus.err != cuDAOError::Success) {
                return cuDAOStatus{scheduler.initStatus};
            }
            scheduler.submitTask(std::move(task));
            return cuDAOStatus{
                cuDAOError::Success,
                __func__
            };
        }
        catch (const std::runtime_error&) {
            cuDAOStatus status{
                cuDAOError::ParameterOverflow,
                __func__
            };
            return status;
        }
        catch (const std::bad_alloc&) {
            cuDAOStatus status{
                cuDAOError::HostAllocationFailed,
                __func__
            };
            return status;
        }
        catch (const std::exception&) {
            cuDAOStatus status{
                cuDAOError::InternalError,
                __func__
            };
            return status;
        }
    }

    template <typename Func, typename... Args>
    std::variant<CudaFuture, cuDAOStatus> launchKernelSync(Func func, dim3 grid, dim3 block, size_t sharedMem,
                                                           Args&&... args) noexcept {
        try {
            auto task = buildTask(func, grid, block, sharedMem, std::forward<Args>(args)...);
            auto promise = std::make_shared<CudaPromise>();
            task.promise = promise;
            auto& scheduler = getDefaultScheduler();
            if (scheduler.initStatus.err != cuDAOError::Success) {
                return cuDAOStatus{scheduler.initStatus};
            }
            scheduler.submitTask(std::move(task));
            return CudaFuture{promise};
        }
        catch (const std::runtime_error&) {
            cuDAOStatus status{
                cuDAOError::ParameterOverflow,
                __func__
            };
            return status;
        }
        catch (const std::bad_alloc&) {
            cuDAOStatus status{
                cuDAOError::HostAllocationFailed,
                __func__
            };
            return status;
        }
        catch (const std::exception&) {
            cuDAOStatus status{
                cuDAOError::InternalError,
                __func__
            };
            return status;
        }
    }

    inline cuDAOStatus deviceSynchronize() noexcept {
        auto& scheduler = getDefaultScheduler();
        if (scheduler.initStatus.err != cuDAOError::Success) {
            return cuDAOStatus{scheduler.initStatus};
        }
        while (!scheduler.idle.load(std::memory_order_acquire)) {
        }
        auto status = scheduler.streamPool.synchronizeAll();
        status.where = __func__;
        return status;
    }

    template <typename T>
    cuDAOStatus sync(T* ptr) noexcept {
        try {
            TaskDescriptor task;
            task.taskType = TaskType::Sync;
            auto promise = std::make_shared<CudaPromise>();
            task.promise = promise;
            task.writeArgs[0] = reinterpret_cast<void*>(ptr);
            task.writeArgsCount = 1;
            auto& scheduler = getDefaultScheduler();
            if (scheduler.initStatus.err != cuDAOError::Success) {
                return cuDAOStatus{scheduler.initStatus};
            }
            scheduler.submitTask(std::move(task));
            CudaFuture{promise}.wait();
            return cuDAOStatus{
                cuDAOError::Success,
                __func__
            };
        }
        catch (const std::exception&) {
            cuDAOStatus status{
                cuDAOError::InternalError,
                __func__
            };
            return status;
        }
    }

    template <typename T>
    cuDAOStatus cuDAOfree(T* ptr) noexcept {
        if (!ptr) {
            return cuDAOStatus{cuDAOError::InvalidPtr, __func__};
        }
        unsigned int memType = 0, isManaged = 0;
        void* memAttr[2] = {&memType, &isManaged};
        CUpointer_attribute attrs[2] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE, CU_POINTER_ATTRIBUTE_IS_MANAGED};
        if (const auto re = cuPointerGetAttributes(2, attrs, memAttr,
                                                   reinterpret_cast<CUdeviceptr>(ptr)); re != CUDA_SUCCESS) {
            return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
        }
        if (isManaged) {
            TaskDescriptor task;
            task.taskType = TaskType::Unregister;
            task.writeArgs[0] = reinterpret_cast<void*>(ptr);
            task.writeArgsCount = 1;
            std::shared_ptr<CudaPromise> promise;
            try {
                promise = std::make_shared<CudaPromise>();
            }
            catch (std::bad_alloc&) {
                return cuDAOStatus{cuDAOError::HostAllocationFailed, __func__};
            }
            task.promise = promise;
            CudaFuture future{promise};
            auto& scheduler = getDefaultScheduler();
            if (scheduler.initStatus.err != cuDAOError::Success) {
                return cuDAOStatus{scheduler.initStatus};
            }
            scheduler.submitTask(std::move(task));
            future.wait();
            const auto res = cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
            if (res != CUDA_SUCCESS) {
                return cuDAOStatus{cuDAOError::CudaDriverError, __func__, res};
            }
        }
        else {
            switch (static_cast<CUmemorytype>(memType)) {
            case CU_MEMORYTYPE_DEVICE:
                {
                    TaskDescriptor task;
                    task.taskType = TaskType::Free;
                    task.writeArgs[0] = reinterpret_cast<void*>(ptr);
                    task.writeArgsCount = 1;
                    auto& scheduler = getDefaultScheduler();
                    if (scheduler.initStatus.err != cuDAOError::Success) {
                        return cuDAOStatus{scheduler.initStatus};
                    }
                    scheduler.submitTask(std::move(task));
                    break;
                }
            case CU_MEMORYTYPE_HOST:
                {
                    const auto res = cuMemFreeHost(reinterpret_cast<void*>(ptr));
                    if (res != CUDA_SUCCESS) {
                        return cuDAOStatus{cuDAOError::CudaDriverError, __func__, res};
                    }
                    break;
                }
            default:
                {
                    return cuDAOStatus{cuDAOError::InvalidPtr, __func__};
                }
            }
        }
        return cuDAOStatus{
            cuDAOError::Success,
            __func__
        };
    }

    template <typename T>
    cuDAOStatus cuDAOMemcpy(T* dst, const T* src, const size_t bytes,
                            const cuDAOMemcpyType memcpyType = cuDAOMemcpyType::Auto) noexcept {
        TaskDescriptor task;
        switch (memcpyType) {
        case cuDAOMemcpyType::HostToDevice:
            task.taskType = TaskType::MemcpyHtoD;
            break;
        case cuDAOMemcpyType::DeviceToHost:
            task.taskType = TaskType::MemcpyDtoH;
            break;
        case cuDAOMemcpyType::DeviceToDevice:
            task.taskType = TaskType::MemcpyDtoD;
            break;
        case cuDAOMemcpyType::Auto:
            {
                CUmemorytype dstType, srcType;
                auto re = cuPointerGetAttribute(&dstType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                                reinterpret_cast<CUdeviceptr>(dst));
                if (re == CUDA_ERROR_INVALID_VALUE) {
                    dstType = CU_MEMORYTYPE_HOST;
                }
                else if (re != CUDA_SUCCESS) {
                    return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
                }
                re = cuPointerGetAttribute(&srcType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                           reinterpret_cast<CUdeviceptr>(src));
                if (re == CUDA_ERROR_INVALID_VALUE) {
                    srcType = CU_MEMORYTYPE_HOST;
                }
                else if (re != CUDA_SUCCESS) {
                    return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
                }
                if (!getMemcpyTaskType(dstType, srcType, task.taskType)) {
                    return cuDAOStatus{cuDAOError::InvalidPtr, __func__};
                }
                break;
            }
        default:
            // Should never be reached
            break;
        }
        task.sharedMem = bytes;
        task.writeArgsCount = 1;
        task.writeArgs[0] = reinterpret_cast<void*>(dst);
        task.readArgsCount = 1;
        task.readArgs[0] = reinterpret_cast<void*>(const_cast<T*>(src));
        auto& scheduler = getDefaultScheduler();
        if (scheduler.initStatus.err != cuDAOError::Success) {
            return cuDAOStatus{scheduler.initStatus};
        }
        scheduler.submitTask(std::move(task));
        return cuDAOStatus{cuDAOError::Success};
    }

    template <typename T>
    std::variant<CudaFuture, cuDAOStatus> cuDAOMemcpySync(T* dst, const T* src, const size_t bytes,
                                                          const cuDAOMemcpyType memcpyType = cuDAOMemcpyType::Auto)
        noexcept {
        TaskDescriptor task;
        std::shared_ptr<CudaPromise> promise;
        try {
            promise = std::make_shared<CudaPromise>();
            task.promise = promise;
        }
        catch (const std::bad_alloc&) {
            return cuDAOStatus{cuDAOError::InternalError, __func__};
        }
        switch (memcpyType) {
        case cuDAOMemcpyType::HostToDevice:
            task.taskType = TaskType::MemcpyHtoD;
            break;
        case cuDAOMemcpyType::DeviceToHost:
            task.taskType = TaskType::MemcpyDtoH;
            break;
        case cuDAOMemcpyType::DeviceToDevice:
            task.taskType = TaskType::MemcpyDtoD;
            break;
        case cuDAOMemcpyType::Auto:
            {
                unsigned int dstIsManaged, srcIsManaged;
                CUmemorytype dstType, srcType;
                auto re = cuPointerGetAttribute(&dstIsManaged, CU_POINTER_ATTRIBUTE_IS_MANAGED,
                                                reinterpret_cast<CUdeviceptr>(dst));
                if (re == CUDA_ERROR_INVALID_VALUE) {
                    dstType = CU_MEMORYTYPE_HOST;
                }
                else if (re == CUDA_SUCCESS) {
                    if (dstIsManaged) {
                        dstType = CU_MEMORYTYPE_UNIFIED;
                    }
                    else {
                        re = cuPointerGetAttribute(&dstType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                                   reinterpret_cast<CUdeviceptr>(dst));
                        if (re != CUDA_SUCCESS) {
                            return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
                        }
                    }
                }
                else {
                    return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
                }
                re = cuPointerGetAttribute(&srcIsManaged, CU_POINTER_ATTRIBUTE_IS_MANAGED,
                                           reinterpret_cast<CUdeviceptr>(src));
                if (re == CUDA_ERROR_INVALID_VALUE) {
                    srcType = CU_MEMORYTYPE_HOST;
                }
                else if (re == CUDA_SUCCESS) {
                    if (srcIsManaged) {
                        srcType = CU_MEMORYTYPE_UNIFIED;
                    }
                    else {
                        re = cuPointerGetAttribute(&srcType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                                   reinterpret_cast<CUdeviceptr>(src));
                        if (re != CUDA_SUCCESS) {
                            return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
                        }
                    }
                }
                else {
                    return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
                }
                if (!getMemcpyTaskType(dstType, srcType, task.taskType)) {
                    return cuDAOStatus{cuDAOError::InvalidPtr, __func__};
                }
                break;
            }
        default:
            // Should never be reached
            break;
        }
        task.sharedMem = bytes;
        task.writeArgsCount = 1;
        task.writeArgs[0] = reinterpret_cast<void*>(dst);
        task.readArgsCount = 1;
        task.readArgs[0] = reinterpret_cast<void*>(const_cast<T*>(src));
        auto& scheduler = getDefaultScheduler();
        if (scheduler.initStatus.err != cuDAOError::Success) {
            return cuDAOStatus{scheduler.initStatus};
        }
        scheduler.submitTask(std::move(task));
        return CudaFuture{promise};
    }

    template <typename T>
    cuDAOStatus cuDAOMalloc(T** ptr, const size_t bytes, const cuDAOMemKind memKind = cuDAOMemKind::Device) noexcept {
        CUresult re;
        TaskDescriptor task;
        task.taskType = TaskType::Register;
        switch (memKind) {
        case cuDAOMemKind::Device:
            re = cuMemAlloc(reinterpret_cast<CUdeviceptr*>(ptr), bytes);
            break;
        case cuDAOMemKind::Host:
            re = cuMemAllocHost(reinterpret_cast<void**>(ptr), bytes);
            break;
        case cuDAOMemKind::Unified:
            re = cuMemAllocManaged(reinterpret_cast<CUdeviceptr*>(ptr), bytes, CU_MEM_ATTACH_GLOBAL);
            break;
        default:
            re = CUDA_ERROR_INVALID_VALUE;
            break;
        }
        if (re != CUDA_SUCCESS) {
            return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
        }
        task.readArgs[0] = reinterpret_cast<void*>(*ptr);
        task.readArgsCount = 1;
        auto& scheduler = getDefaultScheduler();
        if (scheduler.initStatus.err != cuDAOError::Success) {
            return cuDAOStatus{scheduler.initStatus};
        }
        scheduler.submitTask(std::move(task));
        return cuDAOStatus{cuDAOError::Success};
    }

    template <typename T>
    cuDAOStatus cuDAOMallocAsync(T** ptr, const size_t bytes) noexcept {
        TaskDescriptor task;
        task.taskType = TaskType::Alloc;
        task.writeArgs[0] = reinterpret_cast<void*>(ptr);
        task.writeArgsCount = 1;
        task.sharedMem = bytes;
        std::shared_ptr<CudaPromise> promise;
        try {
            promise = std::make_shared<CudaPromise>();
            task.promise = promise;
        }
        catch (const std::bad_alloc&) {
            return cuDAOStatus{cuDAOError::InternalError, __func__};
        }
        CudaFuture future{promise};
        auto& scheduler = getDefaultScheduler();
        if (scheduler.initStatus.err != cuDAOError::Success) {
            return cuDAOStatus{scheduler.initStatus};
        }
        scheduler.submitTask(std::move(task));
        future.wait();
        return cuDAOStatus{cuDAOError::Success};
    }

    template <typename T, typename U>
    cuDAOStatus cuDAOMemset(T* ptr, const U val, const size_t count) noexcept {
        static_assert(sizeof(U) == 1 || sizeof(U) == 2 || sizeof(U) == 4,
                      "cuDAOMemset only supports 1/2/4-byte types");
        TaskDescriptor task;
        task.taskType = TaskType::Memset;
        task.writeArgs[0] = reinterpret_cast<void*>(ptr);
        task.writeArgsCount = 1;
        task.sharedMem = count;
        std::memcpy(task.paramBuffer.data(), &val, sizeof(U));
        task.paramSizes[0] = sizeof(U);
        task.paramCount = 1;
        auto& scheduler = getDefaultScheduler();
        if (scheduler.initStatus.err != cuDAOError::Success) {
            return cuDAOStatus{scheduler.initStatus};
        }
        scheduler.submitTask(std::move(task));
        return cuDAOStatus{cuDAOError::Success};
    }
}