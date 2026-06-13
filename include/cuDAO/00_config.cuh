#pragma once
#ifndef __CUDACC__
#error "cuDAO.cuh must be compiled with nvcc. Include this file only in .cu files."
#endif
#include "version.h"
#include <cuda.h>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <array>
#include <memory>
#include <cstring>
#include <unordered_map>
#include <thread>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace cuDAO {
    // ──────────────────────────────────────────────────────────────────────────
    // Constants
    // ──────────────────────────────────────────────────────────────────────────
    namespace constants {
        inline constexpr size_t QUEUE_CAPACITY = 1024;
        inline constexpr size_t MAX_PARAM_COUNT = 32;
        inline constexpr size_t PARAM_BUFFER_SIZE = MAX_PARAM_COUNT * 8;
        inline constexpr size_t MAX_TRACKED_PTRS = 1024;
        inline constexpr size_t STREAM_COUNT = 16;
        inline constexpr size_t SCHEDULER_SPIN_COUNT = 1000;

        static_assert((QUEUE_CAPACITY & (QUEUE_CAPACITY - 1)) == 0, "QUEUE_CAPACITY must be a power of 2");
        static_assert((MAX_TRACKED_PTRS & (MAX_TRACKED_PTRS - 1)) == 0, "MAX_TRACKED_PTRS must be a power of 2");
        static_assert((STREAM_COUNT & (STREAM_COUNT - 1)) == 0, "STREAM_COUNT must be a power of 2");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Macros
    // ──────────────────────────────────────────────────────────────────────────

#define CUDAO_ASSERT(expr) \
if (const CUresult res_ = (expr); res_ != CUDA_SUCCESS) { \
const char* errStr_ = nullptr; \
cuGetErrorString(res_, &errStr_); \
fprintf(stderr, "[cuDAO] Fatal error in %s: %s\n", #expr, errStr_); \
std::abort(); \
}
}
