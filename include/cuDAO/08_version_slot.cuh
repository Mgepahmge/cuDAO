#pragma once
#include "07_queue.cuh"

namespace cuDAO {

    // ──────────────────────────────────────────────────────────────────────────
    // Version Slot
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * @brief Dependency state assigned to one tracked pointer.
     *
     * A VersionSlot stores host-side metadata for a pointer and indexes into the
     * backing memory owned by VersionSlotPool. The indexed device memory stores the
     * stream-visible write version, while the indexed pinned host memory stores the
     * read gate.
     */
    struct VersionSlot {
        uint32_t slotIndex;
        uint64_t expectedWriteVersion;
        int32_t pendingReads;

        /**
         * @brief Return the device address of this slot's write-version counter.
         *
         * @param base Base device allocation owned by VersionSlotPool.
         * @return Device pointer used with cuStreamWaitValue64/cuStreamWriteValue64.
         */
        [[nodiscard]] CUdeviceptr getWriteVersionAddr(CUdeviceptr base) const noexcept {
            return base + slotIndex * sizeof(uint64_t);
        }

        /**
         * @brief Return the pinned-host address of this slot's read gate.
         *
         * @param base Base pinned host allocation owned by VersionSlotPool.
         * @return Host pointer used as a stream-memory-operation target.
         */
        [[nodiscard]] uint64_t* getReadGateAddr(uint64_t* base) const noexcept {
            return base + slotIndex;
        }
    };

    /**
     * @brief Pool that owns all version slots and their backing memory.
     *
     * The pool allocates device memory for write versions, pinned host memory for
     * read gates, and a fixed array of reusable VersionSlot objects.
     */
    struct VersionSlotPool {
        CUdeviceptr deviceMem;
        uint64_t* pinnedMem;

        std::array<uint32_t, constants::MAX_TRACKED_PTRS> freeSlots;
        uint32_t freeTop;

        std::array<VersionSlot, constants::MAX_TRACKED_PTRS> versionSlots;

        /**
         * @brief Allocate and initialize backing memory for all version slots.
         *
         * @return CUDA_SUCCESS on success, otherwise the CUDA Driver API error code.
         */
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

        /**
         * @brief Release device and pinned host backing memory.
         */
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

        /**
         * @brief Allocate one reusable version slot from the pool.
         *
         * @return Pointer to an available slot, or nullptr if the pool is exhausted.
         *
         * @note This resets host-side slot metadata. The corresponding device-side
         *       write version must also be reset before a freed slot becomes reusable.
         */
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

        /**
         * @brief Return a slot to the pool free list.
         *
         * @param slot Slot previously allocated by alloc().
         */
        void free(const VersionSlot* slot) noexcept {
            freeSlots[freeTop++] = slot->slotIndex;
        }
    };


    // ──────────────────────────────────────────────────────────────────────────
    // Slot Map
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * @brief Hash function for raw pointer keys in the scheduler slot map.
     */
    struct PtrHash {
        /**
         * @brief Hash a pointer value after discarding low alignment bits.
         */
        size_t operator()(void* ptr) const noexcept {
            return reinterpret_cast<size_t>(ptr) >> 8;
        }
    };

    using SlotMapT = std::unordered_map<void*, VersionSlot*, PtrHash>;

    /**
     * @brief Return the global pointer-to-slot map.
     */
    inline SlotMapT& getSlotMap() {
        static SlotMapT slotMap;
        return slotMap;
    }

    /**
     * @brief Remove a pointer from the global slot map and return its slot.
     *
     * @param ptr Pointer to unregister.
     * @param slotPool Pool that owns the pointer's version slot.
     *
     * @note Call this only after stream-ordered waits guarantee no remaining work
     *       can access the slot state for the old pointer.
     */
    inline void unregisterPtr(void* ptr, VersionSlotPool& slotPool) noexcept {
        auto& map = getSlotMap();
        if (const auto it = map.find(ptr); it != map.end()) {
            slotPool.free(it->second);
            map.erase(it);
        }
    }
}