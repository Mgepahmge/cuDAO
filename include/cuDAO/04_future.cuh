#pragma once
#include "03_status.cuh"

namespace cuDAO {
    // ──────────────────────────────────────────────────────────────────────────
    // CUDA Promise & CUDA Future
    // ──────────────────────────────────────────────────────────────────────────

    struct CudaPromise {
        std::atomic<bool> ready{false};
        std::mutex mtx;
        std::condition_variable cv;

        void set() noexcept {
            ready.store(true, std::memory_order_release);
            cv.notify_one();
        }
    };

    class CudaFuture {
        std::shared_ptr<CudaPromise> promise_;

    public:
        explicit CudaFuture(std::shared_ptr<CudaPromise> p) : promise_(std::move(p)) {
        }

        // Exception should never be thrown.
        // If an exception is thrown, it indicates a significant system issue,
        // And the program should terminate.
        void wait() const noexcept {
            if (promise_->ready.load(std::memory_order_acquire)) return;
            std::unique_lock lock(promise_->mtx);
            promise_->cv.wait(lock, [this] {
                return promise_->ready.load(std::memory_order_relaxed);
            });
        }

        [[nodiscard]] bool ready() const noexcept {
            return promise_->ready.load(std::memory_order_acquire);
        }
    };
}
