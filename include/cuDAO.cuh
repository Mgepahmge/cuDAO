#pragma once
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

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
#include <thread>
#include <optional>

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

    inline void platformWait(WakeFlagT& flag) noexcept {
        syscall(SYS_futex, &flag, FUTEX_WAIT_PRIVATE, 0, nullptr, nullptr, 0);
    }

    inline void platformNotify(WakeFlagT& flag) noexcept {
        flag.store(1, std::memory_order_release);
        syscall(SYS_futex, &flag, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
    }
#endif

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

    // ──────────────────────────────────────────────────────────────────────────
    // cuDAO Error
    // ──────────────────────────────────────────────────────────────────────────

    class LazyString {
        char* data{nullptr};
        size_t size_{0};

    public:
        LazyString() noexcept = default;

        explicit LazyString(const std::string& s) noexcept {
            data = new(std::nothrow) char[s.size() + 1];
            if (data) {
                std::memcpy(data, s.c_str(), s.size() + 1);
                size_ = s.size();
            }
        }

        explicit LazyString(const char* s) noexcept {
            if (!s) {
                return;
            }
            size_ = std::strlen(s);
            data = new(std::nothrow) char[size_ + 1];
            if (data) {
                std::memcpy(data, s, size_ + 1);
            }
        }

        LazyString(const LazyString& s) = delete;

        LazyString(LazyString&& s) noexcept : data(s.data), size_(s.size_) {
            s.data = nullptr;
            s.size_ = 0;
        }

        LazyString& operator=(const LazyString& other) = delete;

        LazyString& operator=(LazyString&& other) noexcept {
            if (this != &other) {
                delete[] data;
                data = other.data;
                size_ = other.size_;
                other.data = nullptr;
                other.size_ = 0;
            }
            return *this;
        }

        [[nodiscard]] size_t size() const {
            return size_;
        }

        [[nodiscard]] const char* c_str() const {
            return data ? data : "";
        }
    };

    struct cuDAOStatus {
        cuDAOError err;
        CUresult cudaResult;
        const char* where;
        LazyString msg;

        explicit cuDAOStatus(const cuDAOError err_, const char* where_ = nullptr,
                             const CUresult cudaResult_ = CUDA_SUCCESS) noexcept : err(err_), cudaResult(cudaResult_),
            where(where_) {
            switch (err) {
            case cuDAOError::Success:
                break;
            case cuDAOError::SlotPoolExhausted:
                msg = LazyString{"No more version slot available. Too many concurrent tracked pointers."};
                break;
            case cuDAOError::InvalidPtr:
                msg = LazyString{"Invalid pointer."};
                break;
            case cuDAOError::ParameterOverflow:
                msg = LazyString{"Too many parameters."};
                break;
            case cuDAOError::CudaDriverError:
                msg = LazyString{"CUDA driver error."};
                break;
            case cuDAOError::InternalError:
                msg = LazyString{"Internal error. This may be caused by insufficient memory or other factors"};
                break;
            case cuDAOError::HostAllocationFailed:
                msg = LazyString{"Host allocation failed."};
                break;
            case cuDAOError::SynchronizeFailed:
                break;
            case cuDAOError::InvalidDeviceFunctionSymbol:
                msg = LazyString{"Invalid device function symbol."};
                break;
            default:
                msg = LazyString{"Unknown error."};
            }
        }

        cuDAOStatus() : cuDAOStatus(cuDAOError::Success) {
        }

        cuDAOStatus(const cuDAOStatus& other) noexcept : cuDAOStatus(other.err, other.where, other.cudaResult) {
        }

        cuDAOStatus(cuDAOStatus&& other) noexcept
            : err(other.err), cudaResult(other.cudaResult),
              where(other.where), msg(std::move(other.msg)) {
        }

        cuDAOStatus& operator=(cuDAOStatus&& other) noexcept {
            if (this == &other) return *this;
            err = other.err;
            cudaResult = other.cudaResult;
            where = other.where;
            msg = std::move(other.msg);
            return *this;
        }
    };

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

    // ──────────────────────────────────────────────────────────────────────────
    // Kernel Parser
    // ──────────────────────────────────────────────────────────────────────────

    template <typename T>
    struct ReadWrapper {
        T* ptr;

        explicit ReadWrapper(T* p) noexcept : ptr(p) {
        }
    };

    template <typename T>
    struct WriteWrapper {
        T* ptr;

        explicit WriteWrapper(T* p) noexcept : ptr(p) {
        }
    };

    template <typename T>
    ReadWrapper<T> read(T* ptr) noexcept {
        return ReadWrapper<T>{ptr};
    }

    template <typename T>
    WriteWrapper<T> write(T* ptr) noexcept {
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
        if (desc.paramCount >= constants::MAX_PARAM_COUNT) {
            throw std::runtime_error("Too many parameters.");
        }
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

    template <typename Func, typename... Args>
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
        MPSCQueue() noexcept : head(0), tail(0) {
            for (size_t i = 0; i < Capacity; ++i) {
                slots[i].sequence.store(i, std::memory_order_relaxed);
            }
        }

        bool push(T&& data) noexcept {
            auto pos = tail.fetch_add(1, std::memory_order_relaxed);
            auto& slot = slots[pos & (Capacity - 1)];
            while (slot.sequence.load(std::memory_order_acquire) != pos) {
            }
            slot.data = std::move(data);
            slot.sequence.store(pos + 1, std::memory_order_release);
            return true;
        }

        bool pop(T& data) noexcept {
            auto pos = head.load(std::memory_order_relaxed);
            auto& slot = slots[pos & (Capacity - 1)];
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
    // Lock-Free SPMC Queue
    // ──────────────────────────────────────────────────────────────────────────
    template <typename T>
    class SPMCQueue {
        struct Node {
            T data;
            std::atomic<Node*> next{nullptr};

            explicit Node(T&& data) : data(std::move(data)) {
            }

            Node() = default;
        };

        std::atomic<Node*> head{nullptr};
        Node* tail{nullptr};

    public:
        SPMCQueue() = default;

        bool init() noexcept {
            auto dummy = new(std::nothrow) Node();
            if (!dummy) {
                return false;
            }
            head.store(dummy, std::memory_order_release);
            tail = dummy;
            return true;
        }

        bool push(T&& item) noexcept {
            auto* node = new(std::nothrow) Node(std::move(item));
            if (!node) {
                return false;
            }
            tail->next.store(node, std::memory_order_release);
            tail = node;
            return true;
        }

        bool pop(T& item) noexcept {
            auto* oldHead = head.load(std::memory_order_acquire);
            if (!oldHead) {
                return false;
            }
            while (true) {
                auto* next = oldHead->next.load(std::memory_order_acquire);
                if (!next) {
                    return false;
                }
                if (head.compare_exchange_weak(oldHead, next, std::memory_order_release, std::memory_order_acquire)) {
                    item = std::move(next->data);
                    delete oldHead;
                    return true;
                }
            }
        }

        ~SPMCQueue() {
            if (!head.load(std::memory_order_acquire)) {
                return;
            }
            T dummy;
            while (pop(dummy)) {
            }
            delete head.load(std::memory_order_acquire);
        }
    };

    // ──────────────────────────────────────────────────────────────────────────
    // Global Task Queue
    // ──────────────────────────────────────────────────────────────────────────

    inline MPSCQueue<TaskDescriptor, constants::QUEUE_CAPACITY>& getTaskQueue() noexcept {
        static MPSCQueue<TaskDescriptor, constants::QUEUE_CAPACITY> queue;
        return queue;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Global Error Queue
    // ──────────────────────────────────────────────────────────────────────────

    inline SPMCQueue<cuDAOStatus>& getErrorQueue() noexcept {
        static SPMCQueue<cuDAOStatus> queue;
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

        CUresult init() noexcept {
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

        void destroy() noexcept {
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

    inline void unregisterPtr(void* ptr, VersionSlotPool& slotPool) noexcept {
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
        std::array<CUstream, constants::STREAM_COUNT> streams{};
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

        cuDAOStatus synchronizeAll() noexcept {
            std::unique_ptr<std::vector<std::pair<uint32_t, CUresult>>> status(nullptr);
            for (uint32_t i = 0; i < constants::STREAM_COUNT; ++i) {
                auto res = cuStreamSynchronize(streams[i]);
                if (res != CUDA_SUCCESS) {
                    if (!status) {
                        status = std::make_unique<std::vector<std::pair<uint32_t, CUresult>>>();
                    }
                    status->emplace_back(i, res);
                }
            }
            if (!status) {
                return cuDAOStatus{
                    cuDAOError::Success
                };
            }
            cuDAOStatus result{
                cuDAOError::SynchronizeFailed
            };
            std::string msgString{"Failed to synchronize the following streams:\n"};
            for (auto i = status->begin(); i != status->end(); ++i) {
                msgString += "Stream " + std::to_string(i->first) + " : ";
                const char* cudaErrStr;
                cuGetErrorString(i->second, &cudaErrStr);
                msgString += cudaErrStr;
                msgString += "\n";
            }
            result.msg = LazyString(msgString);
            return result;
        }
    };

    // ──────────────────────────────────────────────────────────────────────────
    // Scheduler
    // ──────────────────────────────────────────────────────────────────────────

    template <typename Policy = RoundRobinPolicy>
    class Scheduler {
        using TaskQueueT = MPSCQueue<TaskDescriptor, constants::QUEUE_CAPACITY>;
        using ErrorQueueT = SPMCQueue<cuDAOStatus>;

        friend cuDAOStatus deviceSynchronize() noexcept;

        StreamPool<Policy> streamPool;
        VersionSlotPool slotPool{};
        SlotMapT* slotMap{nullptr};
        TaskQueueT* taskQueue{nullptr};
        ErrorQueueT* errorQueue{nullptr};
        WakeFlagT wakeFlag{0};
        std::atomic<bool> stopped{false};
        std::thread thread;
        CUdevice device;
        std::atomic<bool> initialized{false};
        std::unordered_map<void*, CUfunction> funcCache;
        std::array<VersionSlot*, constants::MAX_PARAM_COUNT> writeSlotsCache;
        void* kernelParams[constants::MAX_PARAM_COUNT];

        std::atomic<bool> idle{false};

        cuDAOStatus initCudaContext() const noexcept {
            CUcontext ctx;
            auto re = cuDevicePrimaryCtxRetain(&ctx, device);
            if (re != CUDA_SUCCESS) {
                return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
            }
            re = cuCtxSetCurrent(ctx);
            if (re != CUDA_SUCCESS) {
                return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
            }
            return cuDAOStatus{cuDAOError::Success};
        }

        cuDAOStatus initResource() noexcept {
            auto re = streamPool.init();
            if (re != CUDA_SUCCESS) {
                return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
            }
            re = slotPool.init();
            if (re != CUDA_SUCCESS) {
                return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
            }
            try {
                slotMap = &getSlotMap();
                taskQueue = &getTaskQueue();
            }
            catch (const std::bad_alloc&) {
                return cuDAOStatus{cuDAOError::HostAllocationFailed, __func__};
            }
            errorQueue = &getErrorQueue();
            if (!errorQueue->init()) {
                return cuDAOStatus{cuDAOError::HostAllocationFailed, __func__};
            }
            return cuDAOStatus{cuDAOError::Success};
        }

        void destroyCudaContext() const noexcept {
            cuDevicePrimaryCtxRelease(device);
        }

        void destroyResource() noexcept {
            streamPool.destroy();
            slotPool.destroy();
        }

        bool getCudaFunction(void* funcPtr, CUfunction& kernel) noexcept {
            if (const auto it = funcCache.find(funcPtr); it != funcCache.end()) {
                kernel = it->second;
                return true;
            }
            CUfunction func;
            if (const auto re = cudaGetFuncBySymbol(reinterpret_cast<cudaFunction_t*>(&func), funcPtr); re !=
                cudaSuccess) {
                return false;
            }
            funcCache[funcPtr] = func;
            kernel = func;
            return true;
        }

        struct ReadCallbackData {
            std::array<VersionSlot*, constants::MAX_PARAM_COUNT>* readSlots;
            size_t readArgsCount;
            VersionSlotPool* slotPool;
        };

        struct CompletionCallbackData {
            std::array<VersionSlot*, constants::MAX_PARAM_COUNT> readSlots;
            size_t readArgsCount;
            VersionSlotPool* slotPool;
            CudaPromise* promise;
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            LeastTaskPolicy* policy;
            uint32_t streamId;
#endif
        };

        struct SyncCallbackData {
            CudaPromise* promise;
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            LeastTaskPolicy* policy;
            uint32_t streamId;
#endif
        };

        struct FreeCallbackData {
            VersionSlotPool* slotPool;
            void* ptr;
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            LeastTaskPolicy* policy;
            uint32_t streamId;
#endif
        };

        struct UnregisterCallbackData {
            VersionSlotPool* slotPool;
            void* ptr;
            CudaPromise* promise;
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            LeastTaskPolicy* policy;
            uint32_t streamId;
#endif
        };

        static void readStartCallback(void* data) noexcept {
            auto* data_ = reinterpret_cast<ReadCallbackData*>(data);
            auto* slotPool_ = data_->slotPool;
            auto readArgsCount = data_->readArgsCount;

            for (size_t i = 0; i < readArgsCount; ++i) {
                auto slot = data_->readSlots->at(i);
                ++slot->pendingReads;
                *slot->getReadGateAddr(slotPool_->pinnedMem) = 1;
            }

            delete data_;
        }

        static void syncCallback(void* data) noexcept {
            auto* data_ = reinterpret_cast<SyncCallbackData*>(data);
            auto* promise = data_->promise;
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            auto* policy = data_->policy;
            auto streamId = data_->streamId;
            policy->complete(streamId);
#endif
            if (promise) {
                promise->set();
            }
            delete data_;
        }

        static void freeCallback(void* data) noexcept {
            auto* data_ = reinterpret_cast<FreeCallbackData*>(data);
            auto* ptr = data_->ptr;
            auto* slotPool = data_->slotPool;
            unregisterPtr(ptr, *slotPool);
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            auto* policy = data_->policy;
            auto streamId = data_->streamId;
            policy->complete(streamId);
#endif
            delete data_;
        }

        static void unregisterCallback(void* data) noexcept {
            auto* data_ = reinterpret_cast<UnregisterCallbackData*>(data);
            auto* ptr = data_->ptr;
            auto* slotPool = data_->slotPool;
            auto* promise = data_->promise;
            unregisterPtr(ptr, *slotPool);
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            auto* policy = data_->policy;
            auto streamId = data_->streamId;
            policy->complete(streamId);
#endif
            promise->set();
            delete data_;
        }

        static void completionCallBack(void* data) noexcept {
            auto* data_ = reinterpret_cast<CompletionCallbackData*>(data);
            auto* slotPool_ = data_->slotPool;
            auto* promise = data_->promise;
            auto readArgsCount = data_->readArgsCount;

            for (size_t i = 0; i < readArgsCount; ++i) {
                auto slot = data_->readSlots[i];
                if (--slot->pendingReads == 0) {
                    *slot->getReadGateAddr(slotPool_->pinnedMem) = 0;
                }
            }

            if (promise) {
                promise->set();
            }

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            data_->policy->complete(data_->streamId);
#endif

            delete data_;
        }

        void processKernelTask(TaskDescriptor& task) noexcept {
            // Phase 1 : Reversible operations
            // Allocate callback data
            auto* completionData = new(std::nothrow) CompletionCallbackData;
            if (!completionData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }
            completionData->readArgsCount = task.readArgsCount;
            completionData->promise = task.promise.get();
            completionData->slotPool = &slotPool;
            auto* readData = new(std::nothrow) ReadCallbackData{
                &completionData->readSlots,
                task.readArgsCount,
                &slotPool
            };
            if (!readData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                delete completionData;
                return;
            }

            // Register parameters
            for (size_t i = 0; i < task.writeArgsCount; ++i) {
                auto writeArg = task.writeArgs[i];
                auto [it, inserted] = slotMap->try_emplace(writeArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete readData;
                        delete completionData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                writeSlotsCache[i] = slot;
            }
            for (size_t i = 0; i < task.readArgsCount; ++i) {
                auto readArg = task.readArgs[i];
                auto [it, inserted] = slotMap->try_emplace(readArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete readData;
                        delete completionData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                completionData->readSlots[i] = slot;
            }

            // Get kernel
            CUfunction kernel;
            if (!getCudaFunction(task.func, kernel)) {
                delete readData;
                delete completionData;
                errorQueue->push(cuDAOStatus{cuDAOError::InvalidDeviceFunctionSymbol, __func__});
                return;
            }

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            completionData->streamId = streamId;
            completionData->policy = &streamPool.policy;
#endif

            // Wait write version
            for (size_t i = 0; i < task.writeArgsCount; ++i) {
                auto* slot = writeSlotsCache[i];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
                ++slot->expectedWriteVersion;
            }
            for (size_t i = 0; i < task.readArgsCount; ++i) {
                auto* slot = completionData->readSlots[i];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
            }

            // Wait read gate
            for (size_t i = 0; i < task.writeArgsCount; ++i) {
                auto* slot = writeSlotsCache[i];

                CUDAO_ASSERT(cuStreamWaitValue64(
                    stream, reinterpret_cast<CUdeviceptr>(slot->getReadGateAddr(slotPool.pinnedMem)), 0,
                    CU_STREAM_WAIT_VALUE_EQ));
            }

            // Launch read start callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, readStartCallback, readData));

            // Launch kernel
            for (size_t i = 0; i < task.paramCount; ++i) {
                kernelParams[i] = task.paramBuffer.data() + task.paramOffsets[i];
            }
            CUDAO_ASSERT(cuLaunchKernel(kernel,
                task.grid.x, task.grid.y, task.grid.z,
                task.block.x, task.block.y, task.block.z,
                task.sharedMem, stream,
                kernelParams, nullptr));

            // Update write version
            for (size_t i = 0; i < task.writeArgsCount; ++i) {
                auto* slot = writeSlotsCache[i];

                CUDAO_ASSERT(cuStreamWriteValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WRITE_VALUE_DEFAULT));
            }

            // Launch completion callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, completionCallBack, completionData));
        }

        void processSyncTask(TaskDescriptor& task) noexcept {
            auto ptr = task.writeArgs[0];
            auto it = slotMap->find(ptr);
            // Phase 1 : Reversible operations
            // Register pointer
            if (it == slotMap->end()) {
                auto* slot = slotPool.alloc();
                if (!slot) {
                    errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                    return;
                }
                slotMap->emplace(ptr, slot);
                task.promise->set();
                return;
            }
            auto slot = it->second;

            // Allocate callback data
            auto* syncData = new(std::nothrow) SyncCallbackData{
                task.promise.get()
            };
            if (!syncData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            syncData->streamId = streamId;
            syncData->policy = &streamPool.policy;
#endif

            // Wait write version
            CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));

            // Launch sync callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, syncCallback, reinterpret_cast<void*>(syncData)));
        }

        void processFreeTask(TaskDescriptor& task) noexcept {
            auto ptr = task.writeArgs[0];
            auto it = slotMap->find(ptr);
            if (it == slotMap->end()) {
                // Phase 2 : Irreversible operations
                // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
                uint32_t streamId;
                auto stream = streamPool.get(&streamId);
#else
                auto stream = streamPool.get();
#endif

                // Free data
                CUDAO_ASSERT(cuMemFreeAsync(reinterpret_cast<CUdeviceptr>(ptr), stream));
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
                streamPool.policy.complete(streamId);
#endif
                return;
            }

            auto slot = it->second;

            // Phase 1 : Reversible operations
            // Allocate free callback data
            auto* freeData = new(std::nothrow) FreeCallbackData{
                &slotPool,
                ptr
            };
            if (!freeData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }

            // Phase 2 : irreversible operations
            // Get Stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            freeData->streamId = streamId;
            freeData->policy = &streamPool.policy;
#endif

            // Wait write version & read gate
            CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
            CUDAO_ASSERT(cuStreamWaitValue64(
                stream, reinterpret_cast<CUdeviceptr>(slot->getReadGateAddr(slotPool.pinnedMem)), 0,
                CU_STREAM_WAIT_VALUE_EQ));

            // Free data
            CUDAO_ASSERT(cuMemFreeAsync(reinterpret_cast<CUdeviceptr>(ptr), stream));

            // Launch free callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, freeCallback, reinterpret_cast<void*>(freeData)));
        }

        void processMemcpyDtoDTask(TaskDescriptor& task) noexcept {
            // Phase 1 : Reversible operations
            // Allocate callback data
            auto* completionData = new(std::nothrow) CompletionCallbackData;
            if (!completionData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }
            completionData->readArgsCount = task.readArgsCount;
            completionData->promise = task.promise.get();
            completionData->slotPool = &slotPool;
            auto* readData = new(std::nothrow) ReadCallbackData{
                &completionData->readSlots,
                task.readArgsCount,
                &slotPool
            };
            if (!readData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                delete completionData;
                return;
            }

            // Register parameters
            {
                auto writeArg = task.writeArgs[0];
                auto [it, inserted] = slotMap->try_emplace(writeArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete readData;
                        delete completionData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                writeSlotsCache[0] = slot;
            }
            {
                auto readArg = task.readArgs[0];
                auto [it, inserted] = slotMap->try_emplace(readArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete readData;
                        delete completionData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                completionData->readSlots[0] = slot;
            }

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            completionData->streamId = streamId;
            completionData->policy = &streamPool.policy;
#endif

            // Wait write version
            {
                auto* slot = writeSlotsCache[0];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
                ++slot->expectedWriteVersion;
            }
            {
                auto* slot = completionData->readSlots[0];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
            }

            // Wait read gate
            {
                auto* slot = writeSlotsCache[0];

                CUDAO_ASSERT(cuStreamWaitValue64(
                    stream, reinterpret_cast<CUdeviceptr>(slot->getReadGateAddr(slotPool.pinnedMem)), 0,
                    CU_STREAM_WAIT_VALUE_EQ));
            }

            // Launch read start callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, readStartCallback, readData));

            // Memcpy
            CUDAO_ASSERT(
                cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(task.writeArgs[0]),reinterpret_cast<CUdeviceptr>(task.
                    readArgs[0]),task.sharedMem,stream));

            // Update write version
            {
                auto* slot = writeSlotsCache[0];

                CUDAO_ASSERT(cuStreamWriteValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WRITE_VALUE_DEFAULT));
            }

            // Launch completion callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, completionCallBack, completionData));
        }

        void processMemcpyHtoDTask(TaskDescriptor& task) noexcept {
            // Phase 1 : Reversible operations
            // Allocate callback data
            auto* syncData = new(std::nothrow) SyncCallbackData{
                task.promise.get()
            };
            if (!syncData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }

            // Register parameters
            {
                auto writeArg = task.writeArgs[0];
                auto [it, inserted] = slotMap->try_emplace(writeArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete syncData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                writeSlotsCache[0] = slot;
            }

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            syncData->streamId = streamId;
            syncData->policy = &streamPool.policy;
#endif

            // Wait write version
            {
                auto* slot = writeSlotsCache[0];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
                ++slot->expectedWriteVersion;
            }

            // Wait read gate
            {
                auto* slot = writeSlotsCache[0];

                CUDAO_ASSERT(cuStreamWaitValue64(
                    stream, reinterpret_cast<CUdeviceptr>(slot->getReadGateAddr(slotPool.pinnedMem)), 0,
                    CU_STREAM_WAIT_VALUE_EQ));
            }

            // Memcpy
            CUDAO_ASSERT(
                cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(task.writeArgs[0]),task.readArgs[0],task.sharedMem,
                    stream));

            // Update write version
            {
                auto* slot = writeSlotsCache[0];

                CUDAO_ASSERT(cuStreamWriteValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WRITE_VALUE_DEFAULT));
            }

            // Launch sync callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, syncCallback, reinterpret_cast<void*>(syncData)));
        }

        void processMemcpyDtoHTask(TaskDescriptor& task) noexcept {
            // Phase 1 : Reversible operations
            // Allocate callback data
            auto* completionData = new(std::nothrow) CompletionCallbackData;
            if (!completionData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }
            completionData->readArgsCount = task.readArgsCount;
            completionData->promise = task.promise.get();
            completionData->slotPool = &slotPool;
            auto* readData = new(std::nothrow) ReadCallbackData{
                &completionData->readSlots,
                task.readArgsCount,
                &slotPool
            };
            if (!readData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                delete completionData;
                return;
            }

            // Register parameters
            {
                auto readArg = task.readArgs[0];
                auto [it, inserted] = slotMap->try_emplace(readArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete readData;
                        delete completionData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                completionData->readSlots[0] = slot;
            }

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            completionData->streamId = streamId;
            completionData->policy = &streamPool.policy;
#endif

            // Wait write version
            {
                auto* slot = completionData->readSlots[0];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
            }

            // Launch read start callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, readStartCallback, readData));

            // Memcpy
            CUDAO_ASSERT(
                cuMemcpyDtoHAsync(task.writeArgs[0],reinterpret_cast<CUdeviceptr>(task.readArgs[0]),task.sharedMem,
                    stream));

            // Launch completion callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, completionCallBack, completionData));
        }

        void processMemcpyUtoUTask(TaskDescriptor& task) noexcept {
            // Phase 1 : Reversible operations
            // Allocate callback data
            auto* completionData = new(std::nothrow) CompletionCallbackData;
            if (!completionData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }
            completionData->readArgsCount = task.readArgsCount;
            completionData->promise = task.promise.get();
            completionData->slotPool = &slotPool;
            auto* readData = new(std::nothrow) ReadCallbackData{
                &completionData->readSlots,
                task.readArgsCount,
                &slotPool
            };
            if (!readData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                delete completionData;
                return;
            }

            // Register parameters
            {
                auto writeArg = task.writeArgs[0];
                auto [it, inserted] = slotMap->try_emplace(writeArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete readData;
                        delete completionData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                writeSlotsCache[0] = slot;
            }
            {
                auto readArg = task.readArgs[0];
                auto [it, inserted] = slotMap->try_emplace(readArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete readData;
                        delete completionData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                completionData->readSlots[0] = slot;
            }

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            completionData->streamId = streamId;
            completionData->policy = &streamPool.policy;
#endif

            // Wait write version
            {
                auto* slot = writeSlotsCache[0];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
                ++slot->expectedWriteVersion;
            }
            {
                auto* slot = completionData->readSlots[0];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
            }

            // Wait read gate
            {
                auto* slot = writeSlotsCache[0];

                CUDAO_ASSERT(cuStreamWaitValue64(
                    stream, reinterpret_cast<CUdeviceptr>(slot->getReadGateAddr(slotPool.pinnedMem)), 0,
                    CU_STREAM_WAIT_VALUE_EQ));
            }

            // Launch read start callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, readStartCallback, readData));

            // Memcpy
            CUDAO_ASSERT(
                cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(task.writeArgs[0]),reinterpret_cast<CUdeviceptr>(task.
                    readArgs[0]),task.sharedMem,stream));

            // Update write version
            {
                auto* slot = writeSlotsCache[0];

                CUDAO_ASSERT(cuStreamWriteValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WRITE_VALUE_DEFAULT));
            }

            // Launch completion callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, completionCallBack, completionData));
        }

        void processMemcpyHtoUTask(TaskDescriptor& task) noexcept {
            // Phase 1 : Reversible operations
            // Allocate callback data
            auto* syncData = new(std::nothrow) SyncCallbackData{
                task.promise.get()
            };
            if (!syncData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }

            // Register parameters
            {
                auto writeArg = task.writeArgs[0];
                auto [it, inserted] = slotMap->try_emplace(writeArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete syncData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                writeSlotsCache[0] = slot;
            }

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            syncData->streamId = streamId;
            syncData->policy = &streamPool.policy;
#endif

            // Wait write version
            {
                auto* slot = writeSlotsCache[0];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
                ++slot->expectedWriteVersion;
            }

            // Wait read gate
            {
                auto* slot = writeSlotsCache[0];

                CUDAO_ASSERT(cuStreamWaitValue64(
                    stream, reinterpret_cast<CUdeviceptr>(slot->getReadGateAddr(slotPool.pinnedMem)), 0,
                    CU_STREAM_WAIT_VALUE_EQ));
            }

            // Memcpy
            CUDAO_ASSERT(
                cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(task.writeArgs[0]),reinterpret_cast<CUdeviceptr>(task.
                    readArgs[0]),task.sharedMem,stream));

            // Update write version
            {
                auto* slot = writeSlotsCache[0];

                CUDAO_ASSERT(cuStreamWriteValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WRITE_VALUE_DEFAULT));
            }

            // Launch sync callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, syncCallback, reinterpret_cast<void*>(syncData)));
        }

        void processMemcpyUtoHTask(TaskDescriptor& task) noexcept {
            // Phase 1 : Reversible operations
            // Allocate callback data
            auto* completionData = new(std::nothrow) CompletionCallbackData;
            if (!completionData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }
            completionData->readArgsCount = task.readArgsCount;
            completionData->promise = task.promise.get();
            completionData->slotPool = &slotPool;
            auto* readData = new(std::nothrow) ReadCallbackData{
                &completionData->readSlots,
                task.readArgsCount,
                &slotPool
            };
            if (!readData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                delete completionData;
                return;
            }

            // Register parameters
            {
                auto readArg = task.readArgs[0];
                auto [it, inserted] = slotMap->try_emplace(readArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete readData;
                        delete completionData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                completionData->readSlots[0] = slot;
            }

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            completionData->streamId = streamId;
            completionData->policy = &streamPool.policy;
#endif

            // Wait write version
            {
                auto* slot = completionData->readSlots[0];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
            }

            // Launch read start callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, readStartCallback, readData));

            // Memcpy
            CUDAO_ASSERT(
                cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(task.writeArgs[0]),reinterpret_cast<CUdeviceptr>(task.
                    readArgs[0]),task.sharedMem,stream));

            // Launch completion callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, completionCallBack, completionData));
        }

        void processRegisterTask(TaskDescriptor& task) noexcept {
            // Register parameters
            auto readArg = task.readArgs[0];
            auto [it, inserted] = slotMap->try_emplace(readArg, nullptr);
            if (inserted) {
                auto* slot = slotPool.alloc();
                if (!slot) {
                    slotMap->erase(it);
                    errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                    return;
                }
                it->second = slot;
            }
        }

        void processAllocTask(TaskDescriptor& task) noexcept {
            // Phase 1
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            auto* syncData = new(std::nothrow) SyncCallbackData;
            if (!syncData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }
#endif

            // Phase 2
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            syncData->streamId = streamId;
            syncData->policy = &streamPool.policy;
#endif

            CUDAO_ASSERT(cuMemAllocAsync(reinterpret_cast<CUdeviceptr*>(task.writeArgs[0]), task.sharedMem, stream));

            task.promise->set();

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            CUDAO_ASSERT(cuLaunchHostFunc(stream, syncCallback, reinterpret_cast<void*>(syncData)));
#endif
            // Register
            auto* ptr = *reinterpret_cast<void**>(task.writeArgs[0]);
            auto [it, inserted] = slotMap->try_emplace(ptr, nullptr);
            if (inserted) {
                auto* slot = slotPool.alloc();
                if (!slot) {
                    slotMap->erase(it);
                    errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                    return;
                }
                it->second = slot;
            }
            auto* ptrSlot = it->second;

            ptrSlot->expectedWriteVersion = 1;
            CUDAO_ASSERT(cuStreamWriteValue64(stream, ptrSlot->getWriteVersionAddr(slotPool.deviceMem),
                ptrSlot->expectedWriteVersion, CU_STREAM_WRITE_VALUE_DEFAULT));
        }

        void processUnregisterTask(TaskDescriptor& task) noexcept {
            auto ptr = task.writeArgs[0];
            auto it = slotMap->find(ptr);
            if (it == slotMap->end()) {
                task.promise->set();
                return;
            }

            auto slot = it->second;

            // Phase 1 : Reversible operations
            // Allocate free callback data
            auto* unregisterData = new(std::nothrow) UnregisterCallbackData{
                &slotPool,
                ptr,
                task.promise.get()
            };
            if (!unregisterData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }

            // Phase 2 : irreversible operations
            // Get Stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            unregisterData->streamId = streamId;
            unregisterData->policy = &streamPool.policy;
#endif

            // Wait write version & read gate
            CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
            CUDAO_ASSERT(cuStreamWaitValue64(
                stream, reinterpret_cast<CUdeviceptr>(slot->getReadGateAddr(slotPool.pinnedMem)), 0,
                CU_STREAM_WAIT_VALUE_EQ));

            // Launch callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, unregisterCallback, reinterpret_cast<void*>(unregisterData)));
        }

        void processMemsetTask(TaskDescriptor& task) noexcept {
            // Phase 1 : Reversible operations
            // Allocate callback data
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            auto* syncData = new(std::nothrow) SyncCallbackData{
                task.promise.get()
            };
            if (!syncData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }
#endif
            // Register parameters
            auto writeArg = task.writeArgs[0];
            auto [it, inserted] = slotMap->try_emplace(writeArg, nullptr);
            if (inserted) {
                auto* slot = slotPool.alloc();
                if (!slot) {
                    slotMap->erase(it);
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
                    delete syncData;
#endif
                    errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                    return;
                }
                it->second = slot;
            }
            auto slot = it->second;

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            syncData->streamId = streamId;
            syncData->policy = &streamPool.policy;
#endif

            // Wait write version
            CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
            ++slot->expectedWriteVersion;

            // Wait read gate
            CUDAO_ASSERT(cuStreamWaitValue64(
                stream, reinterpret_cast<CUdeviceptr>(slot->getReadGateAddr(slotPool.pinnedMem)), 0,
                CU_STREAM_WAIT_VALUE_EQ));

            // Memset
            switch (task.paramSizes[0]) {
            case 1:
                {
                    uint8_t v;
                    std::memcpy(&v, task.paramBuffer.data(), 1);
                    CUDAO_ASSERT(cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(task.writeArgs[0]), v, task.sharedMem, stream));
                    break;
                }
            case 2:
                {
                    uint16_t v;
                    std::memcpy(&v, task.paramBuffer.data(), 2);
                    CUDAO_ASSERT(cuMemsetD16Async(reinterpret_cast<CUdeviceptr>(task.writeArgs[0]), v, task.sharedMem, stream));
                    break;
                }
            case 4:
                {
                    uint32_t v;
                    std::memcpy(&v, task.paramBuffer.data(), 4);
                    CUDAO_ASSERT(cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(task.writeArgs[0]), v, task.sharedMem, stream));
                    break;
                }
            default:
                {
                    // Should never be reached
                    break;
                }
            }

            // Update write version
            CUDAO_ASSERT(cuStreamWriteValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
    slot->expectedWriteVersion, CU_STREAM_WRITE_VALUE_DEFAULT));

            // Launch callback
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            CUDAO_ASSERT(cuLaunchHostFunc(stream, syncCallback, reinterpret_cast<void*>(syncData)));
#endif
        }

        void processTask(TaskDescriptor& task) noexcept {
            switch (task.taskType) {
            case TaskType::Kernel:
                processKernelTask(task);
                break;
            case TaskType::Sync:
                processSyncTask(task);
                break;
            case TaskType::Free:
                processFreeTask(task);
                break;
            case TaskType::MemcpyDtoD:
                processMemcpyDtoDTask(task);
                break;
            case TaskType::MemcpyDtoH:
                processMemcpyDtoHTask(task);
                break;
            case TaskType::MemcpyHtoD:
                processMemcpyHtoDTask(task);
                break;
            case TaskType::MemcpyHtoU:
                processMemcpyHtoUTask(task);
                break;
            case TaskType::MemcpyUtoH:
                processMemcpyUtoHTask(task);
                break;
            case TaskType::MemcpyUtoU:
                processMemcpyUtoUTask(task);
                break;
            case TaskType::Register:
                processRegisterTask(task);
                break;
            case TaskType::Alloc:
                processAllocTask(task);
                break;
            case TaskType::Unregister:
                processUnregisterTask(task);
                break;
            case TaskType::Memset:
                processMemsetTask(task);
                break;
            default:
                // Should never be reached
                break;
            }
        }

        void run() {
            auto re = initCudaContext();
            if (re.err != cuDAOError::Success) {
                initStatus = cuDAOStatus{re};
                initialized.store(true, std::memory_order_release);
                return;
            }
            re = initResource();
            if (re.err != cuDAOError::Success) {
                initStatus = cuDAOStatus{re};
                initialized.store(true, std::memory_order_release);
                return;
            }

            initialized.store(true, std::memory_order_release);

            while (!stopped.load(std::memory_order_relaxed)) {
                idle.store(false, std::memory_order_relaxed);
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
                    idle.store(true, std::memory_order_release);
                    continue;
                }

                if (taskQueue->pop(task)) {
                    processTask(task);
                    idle.store(true, std::memory_order_release);
                    continue;
                }

                idle.store(true, std::memory_order_release);
                platformWait(wakeFlag);
            }

            destroyResource();
            destroyCudaContext();
        }

    public:
        cuDAOStatus initStatus{cuDAOError::Success};

        explicit Scheduler(const CUdevice device_) : device(device_) {
            thread = std::thread(&Scheduler::run, this);
            while (!initialized.load(std::memory_order_acquire)) {
            }
        }

        ~Scheduler() {
            stopped.store(true);
            platformNotify(wakeFlag);
            if (thread.joinable()) {
                thread.join();
            }
        }

        void submitTask(TaskDescriptor&& task) noexcept {
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
    // Public API
    // ──────────────────────────────────────────────────────────────────────────
    inline cuDAOStatus cuDAOInit() noexcept {
        const auto& scheduler = getDefaultScheduler();
        return cuDAOStatus{scheduler.initStatus};
    }

    inline std::optional<cuDAOStatus> cuDAOGetLastError() noexcept {
        auto& errorQueue = getErrorQueue();
        cuDAOStatus status{};
        if (!errorQueue.pop(status)) {
            return std::nullopt;
        }
        return cuDAOStatus{status};
    }

    template <typename Func, typename... Args>
    cuDAOStatus launchKernel(Func func, dim3 grid, dim3 block, size_t sharedMem, Args&&... args) noexcept {
        try {
            auto task = buildTask(func, grid, block, sharedMem, std::forward<Args>(args)...);
            auto& scheduler = getDefaultScheduler();
            if (scheduler.initStatus.err != cuDAOError::Success) {
                return cuDAOStatus{scheduler.initStatus};
            }
            scheduler.submitTask(std::move(task));
            return cuDAOStatus{
                cuDAOError::Success,
                __func__
            };
        }
        catch (const std::runtime_error&) {
            cuDAOStatus status{
                cuDAOError::ParameterOverflow,
                __func__
            };
            return status;
        }
        catch (const std::bad_alloc&) {
            cuDAOStatus status{
                cuDAOError::HostAllocationFailed,
                __func__
            };
            return status;
        }
        catch (const std::exception&) {
            cuDAOStatus status{
                cuDAOError::InternalError,
                __func__
            };
            return status;
        }
    }

    template <typename Func, typename... Args>
    std::variant<CudaFuture, cuDAOStatus> launchKernelSync(Func func, dim3 grid, dim3 block, size_t sharedMem,
                                                           Args&&... args) noexcept {
        try {
            auto task = buildTask(func, grid, block, sharedMem, std::forward<Args>(args)...);
            auto promise = std::make_shared<CudaPromise>();
            task.promise = promise;
            auto& scheduler = getDefaultScheduler();
            if (scheduler.initStatus.err != cuDAOError::Success) {
                return cuDAOStatus{scheduler.initStatus};
            }
            scheduler.submitTask(std::move(task));
            return CudaFuture{promise};
        }
        catch (const std::runtime_error&) {
            cuDAOStatus status{
                cuDAOError::ParameterOverflow,
                __func__
            };
            return status;
        }
        catch (const std::bad_alloc&) {
            cuDAOStatus status{
                cuDAOError::HostAllocationFailed,
                __func__
            };
            return status;
        }
        catch (const std::exception&) {
            cuDAOStatus status{
                cuDAOError::InternalError,
                __func__
            };
            return status;
        }
    }

    inline cuDAOStatus deviceSynchronize() noexcept {
        auto& scheduler = getDefaultScheduler();
        if (scheduler.initStatus.err != cuDAOError::Success) {
            return cuDAOStatus{scheduler.initStatus};
        }
        while (!scheduler.idle.load(std::memory_order_acquire)) {
        }
        auto status = scheduler.streamPool.synchronizeAll();
        status.where = __func__;
        return status;
    }

    template <typename T>
    cuDAOStatus sync(T* ptr) noexcept {
        try {
            TaskDescriptor task;
            task.taskType = TaskType::Sync;
            auto promise = std::make_shared<CudaPromise>();
            task.promise = promise;
            task.writeArgs[0] = reinterpret_cast<void*>(ptr);
            task.writeArgsCount = 1;
            auto& scheduler = getDefaultScheduler();
            if (scheduler.initStatus.err != cuDAOError::Success) {
                return cuDAOStatus{scheduler.initStatus};
            }
            scheduler.submitTask(std::move(task));
            CudaFuture{promise}.wait();
            return cuDAOStatus{
                cuDAOError::Success,
                __func__
            };
        }
        catch (const std::exception&) {
            cuDAOStatus status{
                cuDAOError::InternalError,
                __func__
            };
            return status;
        }
    }

    template <typename T>
    cuDAOStatus cuDAOfree(T* ptr) noexcept {
        if (!ptr) {
            return cuDAOStatus{cuDAOError::InvalidPtr, __func__};
        }
        unsigned int memType = 0, isManaged = 0;
        void* memAttr[2] = {&memType, &isManaged};
        CUpointer_attribute attrs[2] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE, CU_POINTER_ATTRIBUTE_IS_MANAGED};
        if (const auto re = cuPointerGetAttributes(2, attrs, memAttr,
                                                   reinterpret_cast<CUdeviceptr>(ptr)); re != CUDA_SUCCESS) {
            return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
        }
        if (isManaged) {
            TaskDescriptor task;
            task.taskType = TaskType::Unregister;
            task.writeArgs[0] = reinterpret_cast<void*>(ptr);
            task.writeArgsCount = 1;
            std::shared_ptr<CudaPromise> promise;
            try {
                promise = std::make_shared<CudaPromise>();
            }
            catch (std::bad_alloc&) {
                return cuDAOStatus{cuDAOError::HostAllocationFailed, __func__};
            }
            task.promise = promise;
            CudaFuture future{promise};
            auto& scheduler = getDefaultScheduler();
            if (scheduler.initStatus.err != cuDAOError::Success) {
                return cuDAOStatus{scheduler.initStatus};
            }
            scheduler.submitTask(std::move(task));
            future.wait();
            const auto res = cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
            if (res != CUDA_SUCCESS) {
                return cuDAOStatus{cuDAOError::CudaDriverError, __func__, res};
            }
        }
        else {
            switch (static_cast<CUmemorytype>(memType)) {
            case CU_MEMORYTYPE_DEVICE:
                {
                    TaskDescriptor task;
                    task.taskType = TaskType::Free;
                    task.writeArgs[0] = reinterpret_cast<void*>(ptr);
                    task.writeArgsCount = 1;
                    auto& scheduler = getDefaultScheduler();
                    if (scheduler.initStatus.err != cuDAOError::Success) {
                        return cuDAOStatus{scheduler.initStatus};
                    }
                    scheduler.submitTask(std::move(task));
                    break;
                }
            case CU_MEMORYTYPE_HOST:
                {
                    const auto res = cuMemFreeHost(reinterpret_cast<void*>(ptr));
                    if (res != CUDA_SUCCESS) {
                        return cuDAOStatus{cuDAOError::CudaDriverError, __func__, res};
                    }
                    break;
                }
            default:
                {
                    return cuDAOStatus{cuDAOError::InvalidPtr, __func__};
                }
            }
        }
        return cuDAOStatus{
            cuDAOError::Success,
            __func__
        };
    }

    template <typename T>
    cuDAOStatus cuDAOMemcpy(T* dst, const T* src, const size_t bytes,
                            const cuDAOMemcpyType memcpyType = cuDAOMemcpyType::Auto) noexcept {
        TaskDescriptor task;
        switch (memcpyType) {
        case cuDAOMemcpyType::HostToDevice:
            task.taskType = TaskType::MemcpyHtoD;
            break;
        case cuDAOMemcpyType::DeviceToHost:
            task.taskType = TaskType::MemcpyDtoH;
            break;
        case cuDAOMemcpyType::DeviceToDevice:
            task.taskType = TaskType::MemcpyDtoD;
            break;
        case cuDAOMemcpyType::Auto:
            {
                CUmemorytype dstType, srcType;
                auto re = cuPointerGetAttribute(&dstType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                                reinterpret_cast<CUdeviceptr>(dst));
                if (re == CUDA_ERROR_INVALID_VALUE) {
                    dstType = CU_MEMORYTYPE_HOST;
                }
                else if (re != CUDA_SUCCESS) {
                    return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
                }
                re = cuPointerGetAttribute(&srcType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                           reinterpret_cast<CUdeviceptr>(src));
                if (re == CUDA_ERROR_INVALID_VALUE) {
                    srcType = CU_MEMORYTYPE_HOST;
                }
                else if (re != CUDA_SUCCESS) {
                    return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
                }
                if (!getMemcpyTaskType(dstType, srcType, task.taskType)) {
                    return cuDAOStatus{cuDAOError::InvalidPtr, __func__};
                }
                break;
            }
        default:
            // Should never be reached
            break;
        }
        task.sharedMem = bytes;
        task.writeArgsCount = 1;
        task.writeArgs[0] = reinterpret_cast<void*>(dst);
        task.readArgsCount = 1;
        task.readArgs[0] = reinterpret_cast<void*>(const_cast<T*>(src));
        auto& scheduler = getDefaultScheduler();
        if (scheduler.initStatus.err != cuDAOError::Success) {
            return cuDAOStatus{scheduler.initStatus};
        }
        scheduler.submitTask(std::move(task));
        return cuDAOStatus{cuDAOError::Success};
    }

    template <typename T>
    std::variant<CudaFuture, cuDAOStatus> cuDAOMemcpySync(T* dst, const T* src, const size_t bytes,
                                                          const cuDAOMemcpyType memcpyType = cuDAOMemcpyType::Auto)
        noexcept {
        TaskDescriptor task;
        std::shared_ptr<CudaPromise> promise;
        try {
            promise = std::make_shared<CudaPromise>();
            task.promise = promise;
        }
        catch (const std::bad_alloc&) {
            return cuDAOStatus{cuDAOError::InternalError, __func__};
        }
        switch (memcpyType) {
        case cuDAOMemcpyType::HostToDevice:
            task.taskType = TaskType::MemcpyHtoD;
            break;
        case cuDAOMemcpyType::DeviceToHost:
            task.taskType = TaskType::MemcpyDtoH;
            break;
        case cuDAOMemcpyType::DeviceToDevice:
            task.taskType = TaskType::MemcpyDtoD;
            break;
        case cuDAOMemcpyType::Auto:
            {
                unsigned int dstIsManaged, srcIsManaged;
                CUmemorytype dstType, srcType;
                auto re = cuPointerGetAttribute(&dstIsManaged, CU_POINTER_ATTRIBUTE_IS_MANAGED,
                                                reinterpret_cast<CUdeviceptr>(dst));
                if (re == CUDA_ERROR_INVALID_VALUE) {
                    dstType = CU_MEMORYTYPE_HOST;
                }
                else if (re == CUDA_SUCCESS) {
                    if (dstIsManaged) {
                        dstType = CU_MEMORYTYPE_UNIFIED;
                    }
                    else {
                        re = cuPointerGetAttribute(&dstType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                                   reinterpret_cast<CUdeviceptr>(dst));
                        if (re != CUDA_SUCCESS) {
                            return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
                        }
                    }
                }
                else {
                    return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
                }
                re = cuPointerGetAttribute(&srcIsManaged, CU_POINTER_ATTRIBUTE_IS_MANAGED,
                                           reinterpret_cast<CUdeviceptr>(src));
                if (re == CUDA_ERROR_INVALID_VALUE) {
                    srcType = CU_MEMORYTYPE_HOST;
                }
                else if (re == CUDA_SUCCESS) {
                    if (srcIsManaged) {
                        srcType = CU_MEMORYTYPE_UNIFIED;
                    }
                    else {
                        re = cuPointerGetAttribute(&srcType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                                   reinterpret_cast<CUdeviceptr>(src));
                        if (re != CUDA_SUCCESS) {
                            return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
                        }
                    }
                }
                else {
                    return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
                }
                if (!getMemcpyTaskType(dstType, srcType, task.taskType)) {
                    return cuDAOStatus{cuDAOError::InvalidPtr, __func__};
                }
                break;
            }
        default:
            // Should never be reached
            break;
        }
        task.sharedMem = bytes;
        task.writeArgsCount = 1;
        task.writeArgs[0] = reinterpret_cast<void*>(dst);
        task.readArgsCount = 1;
        task.readArgs[0] = reinterpret_cast<void*>(const_cast<T*>(src));
        auto& scheduler = getDefaultScheduler();
        if (scheduler.initStatus.err != cuDAOError::Success) {
            return cuDAOStatus{scheduler.initStatus};
        }
        scheduler.submitTask(std::move(task));
        return CudaFuture{promise};
    }

    template <typename T>
    cuDAOStatus cuDAOMalloc(T** ptr, const size_t bytes, const cuDAOMemKind memKind = cuDAOMemKind::Device) noexcept {
        CUresult re;
        TaskDescriptor task;
        task.taskType = TaskType::Register;
        switch (memKind) {
        case cuDAOMemKind::Device:
            re = cuMemAlloc(reinterpret_cast<CUdeviceptr*>(ptr), bytes);
            break;
        case cuDAOMemKind::Host:
            re = cuMemAllocHost(reinterpret_cast<void**>(ptr), bytes);
            break;
        case cuDAOMemKind::Unified:
            re = cuMemAllocManaged(reinterpret_cast<CUdeviceptr*>(ptr), bytes, CU_MEM_ATTACH_GLOBAL);
            break;
        default:
            re = CUDA_ERROR_INVALID_VALUE;
            break;
        }
        if (re != CUDA_SUCCESS) {
            return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
        }
        task.readArgs[0] = reinterpret_cast<void*>(*ptr);
        task.readArgsCount = 1;
        auto& scheduler = getDefaultScheduler();
        if (scheduler.initStatus.err != cuDAOError::Success) {
            return cuDAOStatus{scheduler.initStatus};
        }
        scheduler.submitTask(std::move(task));
        return cuDAOStatus{cuDAOError::Success};
    }

    template <typename T>
    cuDAOStatus cuDAOMallocAsync(T** ptr, const size_t bytes) noexcept {
        TaskDescriptor task;
        task.taskType = TaskType::Alloc;
        task.writeArgs[0] = reinterpret_cast<void*>(ptr);
        task.writeArgsCount = 1;
        task.sharedMem = bytes;
        std::shared_ptr<CudaPromise> promise;
        try {
            promise = std::make_shared<CudaPromise>();
            task.promise = promise;
        }
        catch (const std::bad_alloc&) {
            return cuDAOStatus{cuDAOError::InternalError, __func__};
        }
        CudaFuture future{promise};
        auto& scheduler = getDefaultScheduler();
        if (scheduler.initStatus.err != cuDAOError::Success) {
            return cuDAOStatus{scheduler.initStatus};
        }
        scheduler.submitTask(std::move(task));
        future.wait();
        return cuDAOStatus{cuDAOError::Success};
    }

    template <typename T, typename U>
    cuDAOStatus cuDAOMemset(T* ptr, const U val, const size_t count) noexcept {
        static_assert(sizeof(U) == 1 || sizeof(U) == 2 || sizeof(U) == 4,
                      "cuDAOMemset only supports 1/2/4-byte types");
        TaskDescriptor task;
        task.taskType = TaskType::Memset;
        task.writeArgs[0] = reinterpret_cast<void*>(ptr);
        task.writeArgsCount = 1;
        task.sharedMem = count;
        std::memcpy(task.paramBuffer.data(), &val, sizeof(U));
        task.paramSizes[0] = sizeof(U);
        task.paramCount = 1;
        auto& scheduler = getDefaultScheduler();
        if (scheduler.initStatus.err != cuDAOError::Success) {
            return cuDAOStatus{scheduler.initStatus};
        }
        scheduler.submitTask(std::move(task));
        return cuDAOStatus{cuDAOError::Success};
    }
}
