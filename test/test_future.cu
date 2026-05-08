#include <gtest/gtest.h>
#include <cuDAO.cuh>
#include <chrono>
#include <thread>

TEST(cuDAO, CudaFutureWaitUnblocksWhenPromiseSet) {
    auto promise = std::make_shared<cuDAO::CudaPromise>();
    cuDAO::CudaFuture future(promise);

    std::atomic<bool> waitFinished{false};
    std::thread waiter([&future, &waitFinished] {
        future.wait();
        waitFinished.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(waitFinished.load(std::memory_order_acquire));
    EXPECT_FALSE(future.ready());

    promise->set();
    waiter.join();

    EXPECT_TRUE(waitFinished.load(std::memory_order_acquire));
    EXPECT_TRUE(future.ready());
}
