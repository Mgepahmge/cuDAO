#include <gtest/gtest.h>
#include <cuDAO.cuh>
#include <thread>
#include <vector>
#include <algorithm>

TEST(cuDAO, MPSCQueuePushPopSingleElement) {
    cuDAO::MPSCQueue<int, 8> queue;
    ASSERT_TRUE(queue.push(7));

    int out = 0;
    ASSERT_TRUE(queue.pop(out));
    EXPECT_EQ(out, 7);
    EXPECT_FALSE(queue.pop(out));
}

TEST(cuDAO, MPSCQueueSupportsWrapAround) {
    cuDAO::MPSCQueue<int, 8> queue;

    for (int i = 0; i < 8; ++i) {
        ASSERT_TRUE(queue.push(int{i}));
    }

    int out = -1;
    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(queue.pop(out));
        EXPECT_EQ(out, i);
    }

    for (int i = 8; i < 12; ++i) {
        ASSERT_TRUE(queue.push(int{i}));
    }

    for (int i = 4; i < 12; ++i) {
        ASSERT_TRUE(queue.pop(out));
        EXPECT_EQ(out, i);
    }

    EXPECT_FALSE(queue.pop(out));
}

TEST(cuDAO, MPSCQueueAcceptsMultipleProducers) {
    constexpr int producerCount = 4;
    constexpr int itemsPerProducer = 16;
    constexpr int totalItems = producerCount * itemsPerProducer;

    cuDAO::MPSCQueue<int, 128> queue;
    std::vector<std::thread> producers;
    producers.reserve(producerCount);

    for (int p = 0; p < producerCount; ++p) {
        producers.emplace_back([p, &queue] {
            const int base = p * itemsPerProducer;
            for (int i = 0; i < itemsPerProducer; ++i) {
                ASSERT_TRUE(queue.push(base + i));
            }
        });
    }

    for (auto& producer : producers) {
        producer.join();
    }

    std::vector<bool> seen(totalItems, false);
    int out = -1;
    for (int i = 0; i < totalItems; ++i) {
        ASSERT_TRUE(queue.pop(out));
        ASSERT_GE(out, 0);
        ASSERT_LT(out, totalItems);
        seen[out] = true;
    }

    EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](bool value) {
        return value;
    }));
    EXPECT_FALSE(queue.pop(out));
}
