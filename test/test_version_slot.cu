#include <cuda.h>
#include <gtest/gtest.h>
#include <cuDAO.cuh>

TEST(cuDAO, VersionSlotAddressHelpersComputeOffsets) {
    cuDAO::VersionSlot slot{};
    slot.slotIndex = 7;

    constexpr CUdeviceptr baseDevice = 0x1000;
    EXPECT_EQ(
        slot.getWriteVersionAddr(baseDevice),
        baseDevice + slot.slotIndex * sizeof(uint64_t)
    );

    uint64_t gates[32]{};
    EXPECT_EQ(slot.getReadGateAddr(gates), gates + slot.slotIndex);
}

TEST(cuDAO, VersionSlotPoolAllocFreeAndExhaustion) {
    cuDAO::VersionSlotPool pool{};
    pool.freeTop = 3;
    pool.freeSlots[0] = 10;
    pool.freeSlots[1] = 20;
    pool.freeSlots[2] = 30;

    cuDAO::VersionSlot a{};
    cuDAO::VersionSlot b{};
    cuDAO::VersionSlot c{};
    cuDAO::VersionSlot d{};

    ASSERT_TRUE(pool.alloc(a));
    ASSERT_TRUE(pool.alloc(b));
    ASSERT_TRUE(pool.alloc(c));
    EXPECT_FALSE(pool.alloc(d));

    EXPECT_EQ(a.slotIndex, 30u);
    EXPECT_EQ(b.slotIndex, 20u);
    EXPECT_EQ(c.slotIndex, 10u);
    EXPECT_EQ(a.expectedWriteVersion, 0u);
    EXPECT_EQ(b.pendingReads, 0);

    pool.free(b);
    ASSERT_TRUE(pool.alloc(d));
    EXPECT_EQ(d.slotIndex, 20u);
}

TEST(cuDAO, VersionSlotPoolInitAndDestroy) {
    ASSERT_EQ(cuInit(0), CUDA_SUCCESS);

    CUdevice device = 0;
    ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);

    CUcontext ctx = nullptr;
    ASSERT_EQ(cuCtxCreate(&ctx, nullptr, 0, device), CUDA_SUCCESS);

    cuDAO::VersionSlotPool pool{};
    ASSERT_EQ(pool.init(), CUDA_SUCCESS);
    EXPECT_NE(pool.deviceMem, static_cast<CUdeviceptr>(0));
    EXPECT_NE(pool.pinnedMem, nullptr);
    EXPECT_EQ(pool.freeTop, cuDAO::constants::MAX_TRACKED_PTRS);
    EXPECT_EQ(pool.freeSlots[0], 0u);
    EXPECT_EQ(pool.freeSlots[cuDAO::constants::MAX_TRACKED_PTRS - 1], cuDAO::constants::MAX_TRACKED_PTRS - 1);

    pool.destroy();
    EXPECT_EQ(pool.deviceMem, static_cast<CUdeviceptr>(0));
    EXPECT_EQ(pool.pinnedMem, nullptr);

    EXPECT_EQ(cuCtxDestroy(ctx), CUDA_SUCCESS);
}
