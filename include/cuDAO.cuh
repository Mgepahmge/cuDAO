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
#include <cstring>

namespace cuDAO {
    // ──────────────────────────────────────────────────────────────────────────
    // CUDA Promise & CUDA Future
    // ──────────────────────────────────────────────────────────────────────────

    struct CudaPromise {
        std::atomic<bool> ready{false};
        std::mutex mtx;
        std::condition_variable cv;

        void set() {
            ready.store(true, std::memory_order_release);
            cv.notify_one();
        }
    };

    class CudaFuture {
        std::shared_ptr<CudaPromise> promise_;

    public:
        explicit CudaFuture(std::shared_ptr<CudaPromise> p) : promise_(std::move(p)) {
        }

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
        void* func{};
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

        std::shared_ptr<CudaPromise> promise{nullptr};
    };

    // ──────────────────────────────────────────────────────────────────────────
    // Kernel Parser
    // ──────────────────────────────────────────────────────────────────────────

    template <typename T>
    struct ReadWrapper {
        T* ptr;

        explicit ReadWrapper(T* p) : ptr(p) {
        }
    };

    template <typename T>
    struct WriteWrapper {
        T* ptr;

        explicit WriteWrapper(T* p) : ptr(p) {
        }
    };

    template <typename T>
    ReadWrapper<T> read(T* ptr) {
        return ReadWrapper<T>{ptr};
    }

    template <typename T>
    WriteWrapper<T> write(T* ptr) {
        return WriteWrapper<T>{ptr};
    }

    template <typename T>
    struct is_read_wrapper : std::false_type {
    };

    template <typename T>
    struct is_read_wrapper<ReadWrapper<T>> : std::true_type {
    };

    template <typename T>
    struct is_write_wrapper : std::false_type {
    };

    template <typename T>
    struct is_write_wrapper<WriteWrapper<T>> : std::true_type {
    };

    template <typename T>
    struct is_cuda_ptr : std::false_type {
    };

    template <typename T>
    struct is_cuda_ptr<T*> : std::true_type {
    };

    template <typename T>
    struct is_cuda_ptr<const T*> : std::true_type {
    };

    template <typename T>
    void processArg(TaskDescriptor& desc, size_t& offset, T&& arg) {
        using Raw = std::decay_t<T>;
        if constexpr (is_write_wrapper<Raw>::value) {
            auto* ptr = static_cast<void*>(arg.ptr);
            std::memcpy(desc.paramBuffer.data() + offset, &ptr, sizeof(void*));
            desc.paramOffsets[desc.paramCount] = offset;
            desc.paramSizes[desc.paramCount] = sizeof(void*);
            desc.writeArgs[desc.writeArgsCount++] = ptr;
            desc.paramCount++;
            offset += sizeof(void*);
        }
        else if constexpr (is_read_wrapper<Raw>::value) {
            auto* ptr = static_cast<void*>(arg.ptr);
            std::memcpy(desc.paramBuffer.data() + offset, &ptr, sizeof(void*));
            desc.paramOffsets[desc.paramCount] = offset;
            desc.paramSizes[desc.paramCount] = sizeof(void*);
            desc.readArgs[desc.readArgsCount++] = ptr;
            desc.paramCount++;
            offset += sizeof(void*);
        }
        else if constexpr (std::is_pointer_v<Raw>) {
            if constexpr (std::is_const_v<std::remove_pointer_t<Raw>>) {
                // const T* -> void*
                auto* ptr = static_cast<void*>(const_cast<std::remove_const_t<std::remove_pointer_t<Raw>>*>(arg));
                std::memcpy(desc.paramBuffer.data() + offset, &ptr, sizeof(void*));
                desc.paramOffsets[desc.paramCount] = offset;
                desc.paramSizes[desc.paramCount] = sizeof(void*);
                desc.readArgs[desc.readArgsCount++] = ptr;
                desc.paramCount++;
                offset += sizeof(void*);
            }
            else {
                // T* -> void*
                auto* ptr = static_cast<void*>(arg);
                std::memcpy(desc.paramBuffer.data() + offset, &ptr, sizeof(void*));
                desc.paramOffsets[desc.paramCount] = offset;
                desc.paramSizes[desc.paramCount] = sizeof(void*);
                desc.writeArgs[desc.writeArgsCount++] = ptr;
                desc.paramCount++;
                offset += sizeof(void*);
            }
        }
        else {
            Raw val = arg;
            std::memcpy(desc.paramBuffer.data() + offset, &val, sizeof(Raw));
            desc.paramOffsets[desc.paramCount] = offset;
            desc.paramSizes[desc.paramCount] = sizeof(Raw);
            desc.paramCount++;
            offset += sizeof(Raw);
        }
    }

    template<typename Func, typename... Args>
    TaskDescriptor buildTask(Func func, const dim3 grid, const dim3 block, const size_t sharedMem, Args&&... args) {
        TaskDescriptor desc{};
        desc.func = reinterpret_cast<void*>(func);
        desc.grid = grid;
        desc.block = block;
        desc.sharedMem = sharedMem;

        size_t offset = 0;
        (processArg(desc, offset, std::forward<Args>(args)), ...);
        return desc;
    }
}
