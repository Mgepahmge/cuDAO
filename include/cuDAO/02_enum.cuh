#pragma once
#include "01_platform_wait.cuh"

namespace cuDAO {
    // ──────────────────────────────────────────────────────────────────────────
    // Enumeration
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * @brief Internal scheduler task type.
     *
     * TaskType is used by the scheduler to select the execution path for a
     * submitted task descriptor. It is part of the internal scheduling model
     * rather than the normal end-user API.
     */
    enum class TaskType {
        Kernel,      ///< Scheduler-managed CUDA kernel launch.
        Sync,        ///< Pointer-specific synchronization request.
        Free,        ///< Scheduler-managed asynchronous device free.
        MemcpyHtoD,  ///< Host-to-device memory copy.
        MemcpyDtoH,  ///< Device-to-host memory copy.
        MemcpyDtoD,  ///< Device-to-device memory copy.
        MemcpyUtoH,  ///< Unified-memory-to-host memory copy.
        MemcpyUtoU,  ///< Copy involving unified memory on both scheduler-tracked sides.
        MemcpyHtoU,  ///< Host-to-unified-memory memory copy.
        Alloc,       ///< Scheduler-managed asynchronous allocation.
        Register,    ///< Register an externally allocated pointer with the scheduler.
        Unregister,  ///< Remove a tracked pointer from the scheduler.
        Memset,      ///< Scheduler-managed memset task.
        Invalid      ///< Invalid or unsupported task type.
    };

    /**
     * @brief cuDAO-level error code.
     *
     * CUDA Driver API failures are reported as cuDAOError::CudaDriverError and
     * the original CUresult is stored in cuDAOStatus::cudaResult.
     */
    enum class cuDAOError {
        Success = 0,                ///< Operation completed successfully.
        SlotPoolExhausted,          ///< No scheduler version slot is available.
        InvalidPtr,                 ///< The supplied pointer is null, unsupported, or invalid for the requested operation.
        ParameterOverflow,          ///< Too many kernel parameters were packed into a task descriptor.
        CudaDriverError,            ///< A CUDA Driver API call failed; inspect cuDAOStatus::cudaResult.
        InternalError,              ///< Internal cuDAO failure, usually from allocation or unexpected host-side state.
        HostAllocationFailed,       ///< Host-side allocation failed.
        SynchronizeFailed,          ///< Scheduler or stream synchronization failed.
        InvalidDeviceFunctionSymbol ///< The supplied kernel function symbol is not valid for CUDA launch.
    };

    /**
     * @brief Explicit or automatic direction for scheduler-managed memory copy.
     */
    enum class cuDAOMemcpyType {
        HostToDevice,   ///< Copy from host memory to device memory.
        DeviceToHost,   ///< Copy from device memory to host memory.
        DeviceToDevice, ///< Copy between device memory regions.
        Auto            ///< Infer copy direction from CUDA pointer attributes.
    };

    /**
     * @brief Memory kind selected by cuDAOMalloc().
     */
    enum class cuDAOMemKind : uint8_t {
        Host = 0,    ///< CUDA pinned host memory allocated by cuMemAllocHost.
        Device = 1,  ///< Device memory allocated by cuMemAlloc.
        Unified = 2  ///< Managed memory allocated by cuMemAllocManaged.
    };

    // ──────────────────────────────────────────────────────────────────────────
    // Mapping Table
    // ──────────────────────────────────────────────────────────────────────────

    namespace mapping {
        inline constexpr TaskType MemcpyTypeTable[3][3] = {
            // dst/src      Host                   Device                Unified
            /* Host*/ {TaskType::Invalid, TaskType::MemcpyDtoH, TaskType::MemcpyUtoH},
            /* Device*/ {TaskType::MemcpyHtoD, TaskType::MemcpyDtoD, TaskType::MemcpyUtoU},
            /* Unified*/ {TaskType::MemcpyHtoU, TaskType::MemcpyUtoU, TaskType::MemcpyUtoU}
        };
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Utils
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * @brief Convert a CUDA memory type into a cuDAO memory kind.
     *
     * @param type CUDA memory type returned by a pointer attribute query.
     * @param kind Output cuDAO memory kind.
     * @return true if the CUDA memory type is supported by cuDAO.
     */
    static inline bool toMemKind(CUmemorytype type, cuDAOMemKind& kind) noexcept {
        switch (type) {
        case CU_MEMORYTYPE_HOST:
            kind = cuDAOMemKind::Host;
            return true;
        case CU_MEMORYTYPE_DEVICE:
            kind = cuDAOMemKind::Device;
            return true;
        case CU_MEMORYTYPE_UNIFIED:
            kind = cuDAOMemKind::Unified;
            return true;
        default:
            return false;
        }
    }

    /**
     * @brief Map source and destination CUDA memory types to a scheduler memcpy task.
     *
     * @param dstType Destination CUDA memory type.
     * @param srcType Source CUDA memory type.
     * @param taskType Output scheduler task type.
     * @return true if the copy direction is supported.
     */
    static inline bool getMemcpyTaskType(CUmemorytype dstType, CUmemorytype srcType, TaskType& taskType) noexcept {
        cuDAOMemKind dstKind, srcKind;

        if (!toMemKind(dstType, dstKind) || !toMemKind(srcType, srcKind)) {
            taskType = TaskType::Invalid;
            return false;
        }

        taskType = mapping::MemcpyTypeTable[
            static_cast<uint8_t>(dstKind)
        ][
            static_cast<uint8_t>(srcKind)
        ];
        return taskType != TaskType::Invalid;
    }
}
