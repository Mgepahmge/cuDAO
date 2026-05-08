#include <gtest/gtest.h>
#include <cuDAO.cuh>
#include <cstring>

namespace {
    void dummyKernel(int*, const int*, int) {}

    void drainGlobalTaskQueue() {
        auto& queue = cuDAO::getTaskQueue();
        cuDAO::TaskDescriptor discarded{};
        while (queue.pop(discarded)) {}
    }
}

TEST(cuDAO, GlobalTaskQueueIsSingleton) {
    auto& first = cuDAO::getTaskQueue();
    auto& second = cuDAO::getTaskQueue();
    EXPECT_EQ(&first, &second);
}

TEST(cuDAO, LaunchKernelBuildsAndEnqueuesTask) {
    drainGlobalTaskQueue();

    int writeWrapped = 1;
    int readWrapped = 2;
    int scalar = 7;

    cuDAO::launchKernel(
        dummyKernel,
        dim3{2, 1, 1},
        dim3{64, 1, 1},
        16,
        cuDAO::write(&writeWrapped),
        cuDAO::read(&readWrapped),
        scalar
    );

    cuDAO::TaskDescriptor desc{};
    ASSERT_TRUE(cuDAO::getTaskQueue().pop(desc));
    EXPECT_FALSE(cuDAO::getTaskQueue().pop(desc));

    EXPECT_EQ(desc.func, reinterpret_cast<void*>(dummyKernel));
    EXPECT_EQ(desc.grid.x, 2u);
    EXPECT_EQ(desc.block.x, 64u);
    EXPECT_EQ(desc.sharedMem, 16u);
    EXPECT_EQ(desc.paramCount, 3u);
    EXPECT_EQ(desc.writeArgsCount, 1u);
    EXPECT_EQ(desc.readArgsCount, 1u);
    EXPECT_EQ(desc.writeArgs[0], static_cast<void*>(&writeWrapped));
    EXPECT_EQ(desc.readArgs[0], static_cast<void*>(&readWrapped));
    EXPECT_EQ(desc.promise, nullptr);

    int unpackedScalar = 0;
    std::memcpy(
        &unpackedScalar,
        desc.paramBuffer.data() + desc.paramOffsets[2],
        sizeof(unpackedScalar)
    );
    EXPECT_EQ(unpackedScalar, scalar);
}

TEST(cuDAO, LaunchKernelSyncReturnsFutureBoundToTaskPromise) {
    drainGlobalTaskQueue();

    int writeRaw = 9;
    const int readRaw = 3;

    auto future = cuDAO::launchKernelSync(
        dummyKernel,
        dim3{1, 1, 1},
        dim3{32, 1, 1},
        0,
        &writeRaw,
        &readRaw,
        11
    );

    cuDAO::TaskDescriptor desc{};
    ASSERT_TRUE(cuDAO::getTaskQueue().pop(desc));
    EXPECT_FALSE(cuDAO::getTaskQueue().pop(desc));

    ASSERT_NE(desc.promise, nullptr);
    EXPECT_FALSE(future.ready());

    desc.promise->set();
    EXPECT_TRUE(future.ready());

    EXPECT_EQ(desc.writeArgsCount, 1u);
    EXPECT_EQ(desc.readArgsCount, 1u);
    EXPECT_EQ(desc.writeArgs[0], static_cast<void*>(&writeRaw));
    EXPECT_EQ(desc.readArgs[0], static_cast<void*>(const_cast<int*>(&readRaw)));
}
