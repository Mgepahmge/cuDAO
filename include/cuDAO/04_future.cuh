#pragma once
#include "03_status.cuh"

namespace cuDAO {
    // ──────────────────────────────────────────────────────────────────────────
    // CUDA Promise & CUDA Future
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * @brief Shared completion state for scheduler-submitted tasks.
     *
     * CudaPromise is an internal synchronization object used by the scheduler to
     * signal completion to a CudaFuture.
     */
    struct CudaPromise {
        std::atomic<bool> ready{false}; ///< Completion flag.
        std::mutex mtx; ///< Mutex used by the condition variable.
        std::condition_variable cv; ///< Host-side completion notification.

        /**
         * @brief Mark the associated task as complete and notify one waiter.
         */
        void set() noexcept {
            ready.store(true, std::memory_order_release);
            cv.notify_one();
        }
    };

    /**
     * @brief Host-side completion handle for scheduler-submitted work.
     *
     * CudaFuture is returned by APIs that submit work and expose an explicit
     * completion point, such as launchKernelSync() and cuDAOMemcpySync().
     */
    class CudaFuture {
        std::shared_ptr<CudaPromise> promise_;

    public:
        /**
         * @brief Construct a future from a scheduler promise.
         */
        explicit CudaFuture(std::shared_ptr<CudaPromise> p) : promise_(std::move(p)) {
        }

        /**
         * @brief Block the calling host thread until the associated task is complete.
         *
         * This function is noexcept by design. If a standard-library primitive
         * throws here, the failure is treated as a fatal system-level issue.
         */
        void wait() const noexcept {
            if (promise_->ready.load(std::memory_order_acquire)) return;
            std::unique_lock lock(promise_->mtx);
            promise_->cv.wait(lock, [this] {
                return promise_->ready.load(std::memory_order_relaxed);
            });
        }

        /**
         * @brief Check whether the associated task has completed.
         *
         * @return true if the scheduler has signaled completion.
         */
        [[nodiscard]] bool ready() const noexcept {
            return promise_->ready.load(std::memory_order_acquire);
        }
    };
}
