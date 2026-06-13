#pragma once
#include "07_queue.cuh"

namespace cuDAO {

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
}