#include <gtest/gtest.h>
#include <cuDAO.cuh>

namespace {
    void dummyKernel(int*, const int*, int) {}
}

TEST(cuDAO, BuildTaskPacksScalarArguments) {
    auto desc = cuDAO::buildTask(
        dummyKernel,
        dim3{2, 3, 1},
        dim3{4, 5, 1},
        64,
        42u,
        -7
    );

    EXPECT_EQ(desc.func, reinterpret_cast<void*>(dummyKernel));
    EXPECT_EQ(desc.grid.x, 2u);
    EXPECT_EQ(desc.grid.y, 3u);
    EXPECT_EQ(desc.block.x, 4u);
    EXPECT_EQ(desc.block.y, 5u);
    EXPECT_EQ(desc.sharedMem, 64u);
    EXPECT_EQ(desc.paramCount, 2u);
    EXPECT_EQ(desc.readArgsCount, 0u);
    EXPECT_EQ(desc.writeArgsCount, 0u);
    EXPECT_EQ(desc.paramOffsets[0], 0u);
    EXPECT_EQ(desc.paramSizes[0], sizeof(unsigned int));
    EXPECT_EQ(desc.paramOffsets[1], sizeof(unsigned int));
    EXPECT_EQ(desc.paramSizes[1], sizeof(int));

    unsigned int first = 0;
    int second = 0;
    std::memcpy(&first, desc.paramBuffer.data() + desc.paramOffsets[0], sizeof(first));
    std::memcpy(&second, desc.paramBuffer.data() + desc.paramOffsets[1], sizeof(second));
    EXPECT_EQ(first, 42u);
    EXPECT_EQ(second, -7);
}

TEST(cuDAO, BuildTaskClassifiesPointerArguments) {
    int writeWrapped = 0;
    int readWrapped = 1;
    int writeRaw = 2;
    const int readRaw = 3;

    auto desc = cuDAO::buildTask(
        dummyKernel,
        dim3{1, 1, 1},
        dim3{1, 1, 1},
        0,
        cuDAO::write(&writeWrapped),
        cuDAO::read(&readWrapped),
        &writeRaw,
        &readRaw
    );

    EXPECT_EQ(desc.paramCount, 4u);
    EXPECT_EQ(desc.writeArgsCount, 2u);
    EXPECT_EQ(desc.readArgsCount, 2u);
    EXPECT_EQ(desc.writeArgs[0], static_cast<void*>(&writeWrapped));
    EXPECT_EQ(desc.writeArgs[1], static_cast<void*>(&writeRaw));
    EXPECT_EQ(desc.readArgs[0], static_cast<void*>(&readWrapped));
    EXPECT_EQ(desc.readArgs[1], static_cast<void*>(const_cast<int*>(&readRaw)));

    for (size_t i = 0; i < desc.paramCount; ++i) {
        EXPECT_EQ(desc.paramOffsets[i], i * sizeof(void*));
        EXPECT_EQ(desc.paramSizes[i], sizeof(void*));
    }

    void* arg0 = nullptr;
    void* arg1 = nullptr;
    void* arg2 = nullptr;
    void* arg3 = nullptr;
    std::memcpy(&arg0, desc.paramBuffer.data() + desc.paramOffsets[0], sizeof(void*));
    std::memcpy(&arg1, desc.paramBuffer.data() + desc.paramOffsets[1], sizeof(void*));
    std::memcpy(&arg2, desc.paramBuffer.data() + desc.paramOffsets[2], sizeof(void*));
    std::memcpy(&arg3, desc.paramBuffer.data() + desc.paramOffsets[3], sizeof(void*));
    EXPECT_EQ(arg0, static_cast<void*>(&writeWrapped));
    EXPECT_EQ(arg1, static_cast<void*>(&readWrapped));
    EXPECT_EQ(arg2, static_cast<void*>(&writeRaw));
    EXPECT_EQ(arg3, static_cast<void*>(const_cast<int*>(&readRaw)));
}
