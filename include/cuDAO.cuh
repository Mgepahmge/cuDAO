#pragma once

/**
 * @file cuDAO.cuh
 * @brief cuDAO — Dependency-Aware Ordering runtime for concurrent CUDA memory access
 *
 * Single include header. Pull in this file to access the full public API.
 */

#include "cuDAO/version.h"
#include <cuda.h>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <array>
#include <memory>

namespace cuDAO {
    // ──────────────────────────────────────────────────────────────────────────
    // CUDA Promise & CUDA Future
    // ──────────────────────────────────────────────────────────────────────────

    struct CudaPromise {
        std::atomic<bool>       ready{false};
        std::mutex              mtx;
        std::condition_variable cv;

        void set() {
            ready.store(true, std::memory_order_release);
            cv.notify_one();
        }
    };

    class CudaFuture {
        std::shared_ptr<CudaPromise> promise_;
    public:
        explicit CudaFuture(std::shared_ptr<CudaPromise> p) : promise_(std::move(p)) {}

        void wait() const {
            if (promise_->ready.load(std::memory_order_acquire)) return;
            std::unique_lock lock(promise_->mtx);
            promise_->cv.wait(lock, [this] {
                return promise_->ready.load(std::memory_order_relaxed);
            });
        }

        [[nodiscard]] bool ready() const {
            return promise_->ready.load(std::memory_order_acquire);
        }
    };

    // ──────────────────────────────────────────────────────────────────────────
    // Task Descriptor
    // ──────────────────────────────────────────────────────────────────────────

    struct TaskDescriptor {
        CUfunction func{};
        dim3 grid;
        dim3 block;
        size_t sharedMem{};

        std::array<std::byte, 256> paramBuffer{};
        std::array<size_t, 32> paramOffsets{};
        std::array<size_t, 32> paramSizes{};
        size_t paramCount{};

        std::array<void*, 32> writeArgs{};
        std::array<void*, 32> readArgs{};
        size_t writeArgsCount{};
        size_t readArgsCount{};
    };
}
