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
#include <unordered_map>

namespace cuDAO {
    // ──────────────────────────────────────────────────────────────────────────
    // Constants
    // ──────────────────────────────────────────────────────────────────────────
    namespace constants {
        inline constexpr size_t QUEUE_CAPACITY = 1024;
        inline constexpr size_t MAX_PARAM_COUNT = 32;
        inline constexpr size_t PARAM_BUFFER_SIZE = MAX_PARAM_COUNT * 8;
        inline constexpr size_t MAX_TRACKED_PTRS = 1024;

        static_assert((QUEUE_CAPACITY & (QUEUE_CAPACITY - 1)) == 0, "QUEUE_CAPACITY must be a power of 2");
        static_assert((MAX_TRACKED_PTRS & (MAX_TRACKED_PTRS - 1)) == 0, "MAX_TRACKED_PTRS must be a power of 2");
    }


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

    // ──────────────────────────────────────────────────────────────────────────
    // Wait-Free MPSC Queue
    // ──────────────────────────────────────────────────────────────────────────

    template <typename T, size_t Capacity>
    class MPSCQueue {
        static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");

        struct Slot {
            std::atomic<size_t> sequence;
            T data;
        };

        alignas(64) std::array<Slot, Capacity> slots;
        alignas(64) std::atomic<size_t> head;
        alignas(64) std::atomic<size_t> tail;

    public:
        MPSCQueue() : head(0), tail(0) {
            for (size_t i = 0; i < Capacity; ++i) {
                slots[i].sequence.store(i, std::memory_order_relaxed);
            }
        }

        bool push(T&& data) {
            auto pos = tail.fetch_add(1, std::memory_order_relaxed);
            auto& slot = slots[pos & (Capacity-1)];
            while (slot.sequence.load(std::memory_order_acquire) != pos) {}
            slot.data = std::move(data);
            slot.sequence.store(pos + 1, std::memory_order_release);
            return true;
        }

        bool pop(T& data) {
            auto pos = head.load(std::memory_order_relaxed);
            auto& slot = slots[pos & (Capacity-1)];
            if (slot.sequence.load(std::memory_order_acquire) != pos + 1) {
                return false;
            }
            data = std::move(slot.data);
            slot.sequence.store(pos + Capacity, std::memory_order_release);
            head.store(pos + 1, std::memory_order_relaxed);
            return true;
        }
    };

    // ──────────────────────────────────────────────────────────────────────────
    // Global Task Queue
    // ──────────────────────────────────────────────────────────────────────────

    inline MPSCQueue<TaskDescriptor, constants::QUEUE_CAPACITY>& getTaskQueue() {
        static MPSCQueue<TaskDescriptor, constants::QUEUE_CAPACITY> queue;
        return queue;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Kernel Launcher
    // ──────────────────────────────────────────────────────────────────────────

    template<typename Func, typename... Args>
    void launchKernel(Func func, Args&&... args) {
        auto task = buildTask(func, std::forward<Args>(args)...);
        getTaskQueue().push(std::move(task));
    }

    template<typename Func, typename... Args>
    CudaFuture launchKernelSync(Func func, Args&&... args) {
        auto task = buildTask(func, std::forward<Args>(args)...);
        auto promise = std::make_shared<CudaPromise>();
        task.promise = promise;
        getTaskQueue().push(std::move(task));
        return CudaFuture{promise};
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Version Slot
    // ──────────────────────────────────────────────────────────────────────────

    struct VersionSlot {
        uint32_t slotIndex;
        uint64_t expectedWriteVersion;
        int32_t pendingReads;

        [[nodiscard]] CUdeviceptr getWriteVersionAddr(CUdeviceptr base) const noexcept {
            return base + slotIndex * sizeof(uint64_t);
        }

        [[nodiscard]] uint64_t* getReadGateAddr(uint64_t* base) const noexcept {
            return base + slotIndex;
        }
    };

    struct VersionSlotPool {
        CUdeviceptr deviceMem;
        uint64_t* pinnedMem;

        std::array<uint32_t, constants::MAX_TRACKED_PTRS> freeSlots;
        uint32_t freeTop;

        std::array<VersionSlot, constants::MAX_TRACKED_PTRS> versionSlots;

        CUresult init() {
            auto res = cuMemAlloc(&deviceMem, constants::MAX_TRACKED_PTRS * sizeof(uint64_t));

            if (res != CUDA_SUCCESS) {
                return res;
            }

            res = cuMemsetD32(deviceMem, 0, constants::MAX_TRACKED_PTRS * 2);

            if (res != CUDA_SUCCESS) {
                cuMemFree(deviceMem);
                return res;
            }

            res = cuMemAllocHost(reinterpret_cast<void**>(&pinnedMem), constants::MAX_TRACKED_PTRS * sizeof(uint64_t));
            if (res != CUDA_SUCCESS) {
                cuMemFree(deviceMem);
                return res;
            }

            std::memset(pinnedMem, 0, constants::MAX_TRACKED_PTRS * sizeof(uint64_t));

            freeTop = constants::MAX_TRACKED_PTRS;
            for (uint32_t i = 0; i < constants::MAX_TRACKED_PTRS; ++i) {
                freeSlots[i] = i;
            }

            return CUDA_SUCCESS;
        }

        void destroy() {
            if (deviceMem) {
                cuMemFree(deviceMem);
                deviceMem = 0;
            }
            if (pinnedMem) {
                cuMemFreeHost(pinnedMem);
                pinnedMem = nullptr;
            }
        }

        VersionSlot* alloc() noexcept {
            if (freeTop == 0) {
                return nullptr;
            }
            const auto idx = freeSlots[--freeTop];
            auto& slot = versionSlots[idx];
            slot.slotIndex = idx;
            slot.expectedWriteVersion = 0;
            slot.pendingReads = 0;
            return &slot;
        }

        void free(const VersionSlot* slot) noexcept {
            freeSlots[freeTop++] = slot->slotIndex;
        }
    };

    // ──────────────────────────────────────────────────────────────────────────
    // Slot Map
    // ──────────────────────────────────────────────────────────────────────────

    struct PtrHash {
        size_t operator()(void* ptr) const noexcept {
            return reinterpret_cast<size_t>(ptr) >> 8;
        }
    };

    using SlotMapT = std::unordered_map<void*, VersionSlot*, PtrHash>;

    inline SlotMapT& getSlotMap() {
        static SlotMapT slotMap;
        return slotMap;
    }

    inline bool registerPtr(void* ptr, VersionSlotPool& slotPool) {
        auto* slot = slotPool.alloc();
        if (!slot) {
            return false;
        }
        if (!getSlotMap().try_emplace(ptr, slot).second) {
            slotPool.free(slot);
            return false;
        }
        return true;
    }

    inline void unregisterPtr(void* ptr, VersionSlotPool& slotPool) {
        auto& map = getSlotMap();
        if (const auto it = map.find(ptr); it != map.end()) {
            slotPool.free(it->second);
            map.erase(it);
        }
    }
}
