#pragma once
#include "10_Scheduler.cuh"

namespace cuDAO {
    // ──────────────────────────────────────────────────────────────────────────
    // Public API
    // ──────────────────────────────────────────────────────────────────────────
    /**
     * @defgroup public_api Public API
     * @brief Public cuDAO APIs exposed through <cuDAO.cuh>.
     */

    /**
     * @ingroup public_api
     * @brief Initialize the default scheduler and return its initialization status.
     *
     * Most public APIs lazily access the default scheduler, so explicit
     * initialization is optional. Calling cuDAOInit() early is useful when an
     * application wants to detect CUDA context or scheduler initialization
     * failures before submitting work.
     *
     * @return Initialization status of the default scheduler.
     */
    inline cuDAOStatus cuDAOInit() noexcept {
        const auto& scheduler = getDefaultScheduler();
        return cuDAOStatus{scheduler.initStatus};
    }

    /**
     * @ingroup public_api
     * @brief Retrieve the next asynchronous scheduler error, if any.
     *
     * Some errors occur on the scheduler thread after the original API call has
     * returned. Those errors are queued and can be polled with this function.
     *
     * @return The next queued error, or std::nullopt if no error is available.
     */
    inline std::optional<cuDAOStatus> cuDAOGetLastError() noexcept {
        auto& errorQueue = getErrorQueue();
        cuDAOStatus status{};
        if (!errorQueue.pop(status)) {
            return std::nullopt;
        }
        return cuDAOStatus{status};
    }

    /**
     * @ingroup public_api
     * @brief Submit a CUDA kernel launch to the default cuDAO scheduler.
     *
     * Pointer arguments are analyzed to build dependency information. Explicit
     * read() and write() wrappers override type inference. Without wrappers,
     * const pointer arguments are treated as reads and non-const pointer
     * arguments are treated as writes.
     *
     * @tparam Func CUDA global function symbol type.
     * @tparam Args Kernel argument types.
     * @param func CUDA kernel function symbol.
     * @param grid CUDA grid dimensions.
     * @param block CUDA block dimensions.
     * @param sharedMem Dynamic shared memory size in bytes.
     * @param args Kernel arguments.
     * @return Submission status. Success means the task was accepted by the scheduler.
     */
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

    /**
     * @ingroup public_api
     * @brief Submit a CUDA kernel launch and return a host-side completion future.
     *
     * The returned CudaFuture becomes ready after the scheduler has executed the
     * kernel task and released its read dependencies.
     *
     * @tparam Func CUDA global function symbol type.
     * @tparam Args Kernel argument types.
     * @param func CUDA kernel function symbol.
     * @param grid CUDA grid dimensions.
     * @param block CUDA block dimensions.
     * @param sharedMem Dynamic shared memory size in bytes.
     * @param args Kernel arguments.
     * @return CudaFuture on successful submission, otherwise cuDAOStatus.
     */
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

    /**
     * @ingroup public_api
     * @brief Wait until the scheduler is idle and synchronize all scheduler streams.
     *
     * This is a global synchronization point for cuDAO-managed work. Prefer
     * sync(ptr) when only a single dependency target needs to be observed.
     *
     * @return Synchronization status.
     */
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

    /**
     * @ingroup public_api
     * @brief Wait for all previously submitted cuDAO work affecting a pointer.
     *
     * The pointer must be tracked by the scheduler. Device and managed pointers
     * are valid synchronization targets. Ordinary host pointers are not
     * dependency-tracked scheduler slots and should not be passed to sync().
     *
     * @tparam T Pointee type.
     * @param ptr Tracked pointer to synchronize.
     * @return Synchronization status.
     */
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

    /**
     * @ingroup public_api
     * @brief Release CUDA memory through the appropriate cuDAO/CUDA path.
     *
     * Device memory is released through a scheduler-managed asynchronous free
     * task. Managed memory is first unregistered from the scheduler and then
     * released with cuMemFree. CUDA pinned host memory is released with
     * cuMemFreeHost.
     *
     * @tparam T Pointee type.
     * @param ptr Pointer to release.
     * @return Release status.
     */
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

    /**
     * @ingroup public_api
     * @brief Submit a scheduler-managed memory copy.
     *
     * Host pointers may be used as copy endpoints, but ordinary host pointers
     * are not scheduler dependency slots. If the host needs to observe
     * completion of a host-writing copy, use cuDAOMemcpySync().
     *
     * @tparam T Element type used for source and destination pointer typing.
     * @param dst Destination pointer.
     * @param src Source pointer.
     * @param bytes Number of bytes to copy.
     * @param memcpyType Explicit or automatic copy direction.
     * @return Submission status.
     */
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

    /**
     * @ingroup public_api
     * @brief Submit a scheduler-managed memory copy and return a completion future.
     *
     * This is the preferred copy API when the host must explicitly wait for a
     * copy that writes to ordinary host memory, because host pointers are not
     * valid sync(ptr) targets.
     *
     * @tparam T Element type used for source and destination pointer typing.
     * @param dst Destination pointer.
     * @param src Source pointer.
     * @param bytes Number of bytes to copy.
     * @param memcpyType Explicit or automatic copy direction.
     * @return CudaFuture on successful submission, otherwise cuDAOStatus.
     */
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

    /**
     * @ingroup public_api
     * @brief Allocate CUDA memory and register it with the scheduler.
     *
     * Device and unified allocations can participate in scheduler dependency
     * tracking. Host allocations use CUDA pinned host allocation and can be
     * released with cuDAOfree().
     *
     * @tparam T Pointee type.
     * @param ptr Output pointer.
     * @param bytes Number of bytes to allocate.
     * @param memKind Memory kind to allocate.
     * @return Allocation and registration status.
     */
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

    /**
     * @ingroup public_api
     * @brief Submit a scheduler-managed asynchronous device allocation.
     *
     * The function waits until CUDA has assigned a valid virtual address to
     * *ptr. To wait until the allocation is complete before direct external use,
     * call sync(*ptr).
     *
     * @tparam T Pointee type.
     * @param ptr Output pointer location.
     * @param bytes Number of bytes to allocate.
     * @return Allocation submission status.
     */
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

    /**
     * @ingroup public_api
     * @brief Submit a scheduler-managed memset operation.
     *
     * U must be 1, 2, or 4 bytes wide. count is the number of elements of width
     * sizeof(U), matching CUDA Driver memset D8/D16/D32 semantics.
     *
     * @tparam T Destination pointer pointee type.
     * @tparam U Value type; must be 1, 2, or 4 bytes.
     * @param ptr Destination pointer.
     * @param val Value to write.
     * @param count Number of elements to write.
     * @return Submission status.
     */
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

    template <typename T, typename U>
    std::variant<CudaFuture, cuDAOStatus> cuDAOMemsetSync(T* ptr, const U val, const size_t count) noexcept {
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
        std::shared_ptr<CudaPromise> promise;
        try {
            promise = std::make_shared<CudaPromise>();
            task.promise = promise;
        }
        catch (const std::bad_alloc&) {
            return cuDAOStatus{cuDAOError::InternalError, __func__};
        }
        auto& scheduler = getDefaultScheduler();
        if (scheduler.initStatus.err != cuDAOError::Success) {
            return cuDAOStatus{scheduler.initStatus};
        }
        scheduler.submitTask(std::move(task));
        return CudaFuture{promise};
    }
}
