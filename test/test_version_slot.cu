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

    auto* a = pool.alloc();
    auto* b = pool.alloc();
    auto* c = pool.alloc();
    auto* d = pool.alloc();

    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(d, nullptr);

    EXPECT_EQ(a->slotIndex, 30u);
    EXPECT_EQ(b->slotIndex, 20u);
    EXPECT_EQ(c->slotIndex, 10u);
    EXPECT_EQ(a->expectedWriteVersion, 0u);
    EXPECT_EQ(b->pendingReads, 0);

    pool.free(b);
    d = pool.alloc();
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(d->slotIndex, 20u);
}

TEST(cuDAO, RegisterPtrAndUnregisterPtrManageSlotMap) {
    auto& map = cuDAO::getSlotMap();
    map.clear();

    cuDAO::VersionSlotPool pool{};
    pool.freeTop = 2;
    pool.freeSlots[0] = 4;
    pool.freeSlots[1] = 9;

    int a = 1;
    int b = 2;

    ASSERT_TRUE(cuDAO::registerPtr(&a, pool));
    EXPECT_EQ(map.size(), 1u);
    ASSERT_NE(map.find(&a), map.end());
    EXPECT_EQ(map[&a]->slotIndex, 9u);

    EXPECT_FALSE(cuDAO::registerPtr(&a, pool));
    EXPECT_EQ(map.size(), 1u);

    ASSERT_TRUE(cuDAO::registerPtr(&b, pool));
    EXPECT_EQ(map.size(), 2u);
    EXPECT_EQ(map[&b]->slotIndex, 4u);

    cuDAO::unregisterPtr(&a, pool);
    EXPECT_EQ(map.count(&a), 0u);
    EXPECT_EQ(pool.freeTop, 1u);
    EXPECT_EQ(pool.freeSlots[0], 9u);

    cuDAO::unregisterPtr(&b, pool);
    EXPECT_TRUE(map.empty());
    EXPECT_EQ(pool.freeTop, 2u);
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
