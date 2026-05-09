#include <cuda.h>
#include <gtest/gtest.h>
#include <cuDAO.cuh>

TEST(cuDAO, RoundRobinPolicyCyclesAcrossPowerOfTwoStreams) {
    cuDAO::RoundRobinPolicy policy{};
    constexpr uint32_t streamCount = 8;

    for (uint32_t i = 0; i < 2 * streamCount; ++i) {
        EXPECT_EQ(policy.select(streamCount), i & (streamCount - 1));
    }
}

TEST(cuDAO, LeastTaskPolicyPrefersLeastLoadedStreamAndSupportsComplete) {
    cuDAO::LeastTaskPolicy policy{};
    constexpr uint32_t streamCount = 4;

    EXPECT_EQ(policy.select(streamCount), 0u);
    EXPECT_EQ(policy.select(streamCount), 1u);
    EXPECT_EQ(policy.select(streamCount), 2u);
    EXPECT_EQ(policy.select(streamCount), 3u);
    EXPECT_EQ(policy.select(streamCount), 0u);

    policy.complete(0);
    EXPECT_EQ(policy.select(streamCount), 0u);
}

TEST(cuDAO, StreamPoolInitGetAndDestroy) {
    ASSERT_EQ(cuInit(0), CUDA_SUCCESS);

    CUdevice device = 0;
    ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);

    CUcontext ctx = nullptr;
    ASSERT_EQ(cuCtxCreate(&ctx, nullptr, 0, device), CUDA_SUCCESS);

    cuDAO::StreamPool<> pool{};
    ASSERT_EQ(pool.init(), CUDA_SUCCESS);

    uint32_t idx0 = 0;
    uint32_t idx1 = 0;
    CUstream s0 = pool.get(&idx0);
    CUstream s1 = pool.get(&idx1);

    EXPECT_NE(s0, nullptr);
    EXPECT_NE(s1, nullptr);
    EXPECT_EQ(idx0, 0u);
    EXPECT_EQ(idx1, 1u);
    EXPECT_NE(s0, s1);

    pool.destroy();
    EXPECT_EQ(cuCtxDestroy(ctx), CUDA_SUCCESS);
}
