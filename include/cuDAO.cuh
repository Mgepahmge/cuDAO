#pragma once

#ifndef __CUDACC__
#error "cuDAO.cuh must be compiled with nvcc. Include this file only in .cu files."
#endif

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
        inline constexpr size_t STREAM_COUNT = 16;
        inline constexpr size_t SCHEDULER_SPIN_COUNT = 1000;

        static_assert((QUEUE_CAPACITY & (QUEUE_CAPACITY - 1)) == 0, "QUEUE_CAPACITY must be a power of 2");
        static_assert((MAX_TRACKED_PTRS & (MAX_TRACKED_PTRS - 1)) == 0, "MAX_TRACKED_PTRS must be a power of 2");
        static_assert((STREAM_COUNT & (STREAM_COUNT - 1)) == 0, "STREAM_COUNT must be a power of 2");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Platform-specific futex / WaitOnAddress wrapper
    // ──────────────────────────────────────────────────────────────────────────

#ifdef _WIN32
#include <windows.h>

    using WakeFlagT = std::atomic<bool>;

    inline void platformWait(WakeFlagT& flag) {
        bool expected = false;
        WaitOnAddress(&flag, &expected, sizeof(bool), INFINITE);
    }

    inline void platformNotify(WakeFlagT& flag) {
        flag.store(true, std::memory_order_release);
        WakeByAddressAll(&flag);
    }

#else
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

    using WakeFlagT = std::atomic<int32_t>;

    inline void platformWait(WakeFlagT& flag) {
        syscall(SYS_futex, &flag, FUTEX_WAIT_PRIVATE, 0, nullptr, nullptr, 0);
    }

    inline void platformNotify(WakeFlagT& flag) {
        flag.store(1, std::memory_order_release);
        syscall(SYS_futex, &flag, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
    }
#endif


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

    // ──────────────────────────────────────────────────────────────────────────
    // Stream Pool
    // ──────────────────────────────────────────────────────────────────────────

    struct RoundRobinPolicy {
        uint32_t counter = 0;

        uint32_t select(const uint32_t streamCount) noexcept {
            return counter++ & (streamCount - 1);
        }
    };

    struct LeastTaskPolicy {
        std::array<std::atomic<uint32_t>, constants::STREAM_COUNT> taskCount{};

        uint32_t select(uint32_t streamCount) noexcept {
            uint32_t minIdx = 0;
            for (uint32_t i = 1; i < streamCount; ++i) {
                if (taskCount[i].load(std::memory_order_relaxed) < taskCount[minIdx].load(std::memory_order_relaxed)) {
                    minIdx = i;
                }
            }
            taskCount[minIdx].fetch_add(1, std::memory_order_relaxed);
            return minIdx;
        }

        void complete(uint32_t streamIdx) noexcept {
            taskCount[streamIdx].fetch_sub(1, std::memory_order_relaxed);
        }
    };

    template <typename Policy = RoundRobinPolicy>
    struct StreamPool {
        std::array<CUstream, constants::STREAM_COUNT> streams;
        Policy policy;

        CUresult init() noexcept {
            for (uint32_t i = 0; i < constants::STREAM_COUNT; ++i) {
                CUresult res = cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING);
                if (res != CUDA_SUCCESS) {
                    for (uint32_t j = 0; j < i; ++j)
                        cuStreamDestroy(streams[j]);
                    return res;
                }
            }
            return CUDA_SUCCESS;
        }

        void destroy() noexcept {
            for (auto& stream : streams)
                cuStreamDestroy(stream);
        }

        CUstream get(uint32_t* outIdx = nullptr) noexcept {
            uint32_t idx = policy.select(constants::STREAM_COUNT);
            if (outIdx) *outIdx = idx;
            return streams[idx];
        }
    };

    // ──────────────────────────────────────────────────────────────────────────
    // Scheduler
    // ──────────────────────────────────────────────────────────────────────────

    template<typename Policy = RoundRobinPolicy>
    class Scheduler {
        using TaskQueueT = MPSCQueue<TaskDescriptor, constants::QUEUE_CAPACITY>;

        StreamPool<Policy> streamPool;
        VersionSlotPool slotPool{};
        SlotMapT* slotMap{nullptr};
        TaskQueueT* taskQueue{nullptr};
        WakeFlagT wakeFlag{0};
        std::atomic<bool> stopped{false};
        std::thread thread;
        CUdevice device;
        std::atomic<bool> initialized{false};
        std::unordered_map<void*, CUfunction> funcCache;

        void initCudaContext() const {
            CUcontext ctx;
            cuDevicePrimaryCtxRetain(&ctx, device);
            cuCtxSetCurrent(ctx);
        }

        void initResource() {
            streamPool.init();
            slotPool.init();
            slotMap = &getSlotMap();
            taskQueue = &getTaskQueue();
        }

        void destroyCudaContext() const {
            cuDevicePrimaryCtxRelease(device);
        }

        void destroyResource() {
            streamPool.destroy();
            slotPool.destroy();
        }

        CUfunction getCudaFunction(void* funcPtr) {
            if (const auto it = funcCache.find(funcPtr); it != funcCache.end()) {
                return it->second;
            }
            CUfunction func;
            cudaGetFuncBySymbol(reinterpret_cast<cudaFunction_t*>(&func), funcPtr);
            funcCache[funcPtr] = func;
            return func;
        }

        struct ReadCallbackData {
            std::array<void*, constants::MAX_PARAM_COUNT> readArgs;
            size_t readArgsCount;
            VersionSlotPool* slotPool;
        };

        struct CompletionCallbackData {
            std::array<void*, constants::MAX_PARAM_COUNT> readArgs;
            size_t readArgsCount;
            VersionSlotPool* slotPool;
            CudaPromise* promise;
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            LeastTaskPolicy* policy;
            uint32_t streamId;
#endif
        };

        static void readStartCallback(void* data) {
            auto* data_ = reinterpret_cast<ReadCallbackData*>(data);
            auto& readArgs = data_->readArgs;
            auto* slotPool_ = data_->slotPool;
            auto& slotMap_ = getSlotMap();
            auto readArgsCount = data_->readArgsCount;

            for (size_t i = 0; i < readArgsCount; ++i) {
                auto slot = slotMap_.at(readArgs[i]);
                ++slot->pendingReads;
                *slot->getReadGateAddr(slotPool_->pinnedMem) = 1;
            }

            delete data_;
        }

        static void completionCallBack(void* data) {
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            auto* data_ = reinterpret_cast<CompletionCallbackData*>(data);
            auto& readArgs = data_->readArgs;
            auto* slotPool_ = data_->slotPool;
            auto* promise = data_->promise;
            auto* policy = data_->policy;
            auto streamId = data_->streamId;
            auto& slotMap_ = getSlotMap();
            auto readArgsCount = data_->readArgsCount;

            for (size_t i = 0; i < readArgsCount; ++i) {
                auto slot = slotMap_.at(readArgs[i]);
                if (--slot->pendingReads == 0) {
                    *slot->getReadGateAddr(slotPool_->pinnedMem) = 0;
                }
            }

            if (promise) {
                promise->set();
            }

            policy->complete(streamId);

            delete data_;

#else
            auto* data_ = reinterpret_cast<CompletionCallbackData*>(data);
            auto& readArgs = data_->readArgs;
            auto* slotPool_ = data_->slotPool;
            auto* promise = data_->promise;
            auto& slotMap_ = getSlotMap();
            auto readArgsCount = data_->readArgsCount;

            for (size_t i = 0; i < readArgsCount; ++i) {
                auto slot = slotMap_.at(readArgs[i]);
                if (--slot->pendingReads == 0) {
                    *slot->getReadGateAddr(slotPool_->pinnedMem) = 0;
                }
            }

            if (promise) {
                promise->set();
            }

            delete data_;
#endif
        }

        void processTask(TaskDescriptor& task) {
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto& stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

            for (size_t i = 0; i < task.writeArgsCount; ++i) {
                auto writeArg = task.writeArgs[i];
                if (slotMap->find(writeArg) == slotMap->end()) {
                    registerPtr(writeArg, slotPool);
                }
                auto slot = slotMap->at(writeArg);

                cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem), slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ);

                ++slot->expectedWriteVersion;
            }

            for (size_t i = 0; i < task.readArgsCount; ++i) {
                auto readArg = task.readArgs[i];
                if (slotMap->find(readArg) == slotMap->end()) {
                    registerPtr(readArg, slotPool);
                }
                auto slot = slotMap->at(readArg);

                cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem), slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ);
            }

            for (size_t i = 0; i < task.writeArgsCount; ++i) {
                auto writeArg = task.writeArgs[i];
                auto slot = slotMap->at(writeArg);

                cuStreamWaitValue64(stream, reinterpret_cast<CUdeviceptr>(slot->getReadGateAddr(slotPool.pinnedMem)), 0, CU_STREAM_WAIT_VALUE_EQ);
            }

            auto* readData = new ReadCallbackData{
                task.readArgs,
                task.readArgsCount,
                &slotPool
            };

            cuLaunchHostFunc(stream, readStartCallback, readData);

            auto kernel = getCudaFunction(task.func);
            void* kernelParams[constants::MAX_PARAM_COUNT];
            for (size_t i = 0; i < task.paramCount; ++i) {
                kernelParams[i] = task.paramBuffer.data() + task.paramOffsets[i];
            }
            cuLaunchKernel(kernel,
                task.grid.x, task.grid.y, task.grid.z,
                task.block.x, task.block.y, task.block.z,
                task.sharedMem, stream,
                kernelParams, nullptr);

            for (size_t i = 0; i < task.writeArgsCount; ++i) {
                auto writeArg = task.writeArgs[i];
                auto slot = slotMap->at(writeArg);

                cuStreamWriteValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem), slot->expectedWriteVersion, CU_STREAM_WRITE_VALUE_DEFAULT);
            }

            auto* completionData = new CompletionCallbackData{
                task.readArgs,
                task.readArgsCount,
                &slotPool,
                task.promise.get()
            #ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
                , &streamPool.policy, streamId
            #endif
            };
            cuLaunchHostFunc(stream, completionCallBack, completionData);
        }

        void run() {
            initCudaContext();
            initResource();

            initialized.store(true, std::memory_order_release);

            while (!stopped.load(std::memory_order_relaxed)) {
                TaskDescriptor task;

                while (taskQueue->pop(task)) {
                    processTask(task);
                }

                bool found = false;
                for (auto i = 0; i < constants::SCHEDULER_SPIN_COUNT && !found; ++i) {
                    if (taskQueue->pop(task)) {
                        processTask(task);
                        found = true;
                    }
                }
                if (found) {
                    continue;
                }

                if (taskQueue->pop(task)) {
                    processTask(task);
                    continue;
                }

                platformWait(wakeFlag);
            }

            destroyResource();
            destroyCudaContext();
        }

    public:

        explicit Scheduler(const CUdevice device_) : device(device_) {
            thread = std::thread(&Scheduler::run, this);
            while (!initialized.load(std::memory_order_acquire)) {}
        }

        ~Scheduler() {
            stopped.store(true);
            platformNotify(wakeFlag);
            if (thread.joinable()) {
                thread.join();
            }
        }

        void submitTask(TaskDescriptor&& task) {
            taskQueue->push(std::move(task));
            platformNotify(wakeFlag);
        }
    };

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
    using DefaultScheduler = Scheduler<LeastTaskPolicy>;
#else
    using DefaultScheduler = Scheduler<RoundRobinPolicy>;
#endif

    inline DefaultScheduler& getDefaultScheduler() {
        static DefaultScheduler scheduler(0);
        return scheduler;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Kernel Launcher
    // ──────────────────────────────────────────────────────────────────────────

    template<typename Func, typename... Args>
    void launchKernel(Func func, dim3 grid, dim3 block, size_t sharedMem, Args&&... args) {
        auto task = buildTask(func, grid, block, sharedMem, std::forward<Args>(args)...);
        getDefaultScheduler().submitTask(std::move(task));
    }

    template<typename Func, typename... Args>
    CudaFuture launchKernelSync(Func func, dim3 grid, dim3 block, size_t sharedMem, Args&&... args) {
        auto task = buildTask(func, grid, block, sharedMem, std::forward<Args>(args)...);
        auto promise = std::make_shared<CudaPromise>();
        task.promise = promise;
        getDefaultScheduler().submitTask(std::move(task));
        return CudaFuture{promise};
    }
}
