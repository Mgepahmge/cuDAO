#pragma once
#include "00_config.cuh"

namespace cuDAO {

    // ──────────────────────────────────────────────────────────────────────────
    // Platform-specific futex / WaitOnAddress wrapper
    // ──────────────────────────────────────────────────────────────────────────

#ifdef _WIN32
#include <windows.h>

    using WakeFlagT = std::atomic<bool>;

    inline void platformWait(WakeFlagT& flag) {
        bool expected = false;
        WaitOnAddress(&flag, &expected, sizeof(bool), INFINITE);
    }

    inline void platformNotify(WakeFlagT& flag) {
        flag.store(true, std::memory_order_release);
        WakeByAddressAll(&flag);
    }

#else
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

    using WakeFlagT = std::atomic<int32_t>;

    inline void platformWait(WakeFlagT& flag) noexcept {
        syscall(SYS_futex, &flag, FUTEX_WAIT_PRIVATE, 0, nullptr, nullptr, 0);
    }

    inline void platformNotify(WakeFlagT& flag) noexcept {
        flag.store(1, std::memory_order_release);
        syscall(SYS_futex, &flag, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
    }
#endif

}