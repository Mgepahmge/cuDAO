#include <gtest/gtest.h>
#include <cuDAO.cuh>
#include <atomic>
#include <chrono>
#include <thread>

#ifdef _WIN32
TEST(cuDAO, PlatformWaitNotifyUnblocksWaitingThread) {
    std::atomic<bool> flag{false};
    std::atomic<bool> waiterStarted{false};
    std::atomic<bool> waiterFinished{false};

    std::thread waiter([&] {
        waiterStarted.store(true, std::memory_order_release);
        cuDAO::platformWait(flag);
        waiterFinished.store(true, std::memory_order_release);
    });

    while (!waiterStarted.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(waiterFinished.load(std::memory_order_acquire));
    EXPECT_FALSE(flag.load(std::memory_order_acquire));

    cuDAO::platformNotify(flag);
    waiter.join();

    EXPECT_TRUE(flag.load(std::memory_order_acquire));
    EXPECT_TRUE(waiterFinished.load(std::memory_order_acquire));
}
#else
TEST(cuDAO, PlatformWaitNotifyUnblocksWaitingThread) {
    std::atomic<int32_t> flag{0};
    std::atomic<bool> waiterStarted{false};
    std::atomic<bool> waiterFinished{false};

    std::thread waiter([&] {
        waiterStarted.store(true, std::memory_order_release);
        cuDAO::platformWait(flag);
        waiterFinished.store(true, std::memory_order_release);
    });

    while (!waiterStarted.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(waiterFinished.load(std::memory_order_acquire));
    EXPECT_EQ(flag.load(std::memory_order_acquire), 0);

    cuDAO::platformNotify(flag);
    waiter.join();

    EXPECT_EQ(flag.load(std::memory_order_acquire), 1);
    EXPECT_TRUE(waiterFinished.load(std::memory_order_acquire));
}
#endif

