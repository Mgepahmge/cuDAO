#pragma once
#include "04_future.cuh"

namespace cuDAO {

    // ──────────────────────────────────────────────────────────────────────────
    // Task Descriptor
    // ──────────────────────────────────────────────────────────────────────────

    struct TaskDescriptor {
        TaskType taskType{TaskType::Kernel};

        void* func{};
        dim3 grid;
        dim3 block;
        size_t sharedMem{};

        std::array<std::byte, constants::PARAM_BUFFER_SIZE> paramBuffer{};
        std::array<size_t, constants::MAX_PARAM_COUNT> paramOffsets{};
        std::array<size_t, constants::MAX_PARAM_COUNT> paramSizes{};
        size_t paramCount{};

        std::array<void*, constants::MAX_PARAM_COUNT> writeArgs{};
        std::array<void*, constants::MAX_PARAM_COUNT> readArgs{};
        size_t writeArgsCount{};
        size_t readArgsCount{};

        std::shared_ptr<CudaPromise> promise{nullptr};
    };
}