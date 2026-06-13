#pragma once
#include "01_platform_wait.cuh"

namespace cuDAO {
    // ──────────────────────────────────────────────────────────────────────────
    // Enumeration
    // ──────────────────────────────────────────────────────────────────────────

    enum class TaskType {
        Kernel,
        Sync,
        Free,
        MemcpyHtoD,
        MemcpyDtoH,
        MemcpyDtoD,
        MemcpyUtoH,
        MemcpyUtoU,
        MemcpyHtoU,
        Alloc,
        Register,
        Unregister,
        Memset,
        Invalid
    };

    enum class cuDAOError {
        Success = 0,
        SlotPoolExhausted,
        InvalidPtr,
        ParameterOverflow,
        CudaDriverError,
        InternalError,
        HostAllocationFailed,
        SynchronizeFailed,
        InvalidDeviceFunctionSymbol
    };

    enum class cuDAOMemcpyType {
        HostToDevice,
        DeviceToHost,
        DeviceToDevice,
        Auto
    };

    enum class cuDAOMemKind : uint8_t {
        Host = 0,
        Device = 1,
        Unified = 2
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
