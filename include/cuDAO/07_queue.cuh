#pragma once
#include "06_kernel_parser.cuh"

namespace cuDAO {

    // ──────────────────────────────────────────────────────────────────────────
    // Wait-Free MPSC Queue
    // ──────────────────────────────────────────────────────────────────────────

    template <typename T, size_t Capacity>
    class MPSCQueue {
        static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");

        struct Slot {
            std::atomic<size_t> sequence;
            T data;
        };

        alignas(64) std::array<Slot, Capacity> slots;
        alignas(64) std::atomic<size_t> head;
        alignas(64) std::atomic<size_t> tail;

    public:
        MPSCQueue() noexcept : head(0), tail(0) {
            for (size_t i = 0; i < Capacity; ++i) {
                slots[i].sequence.store(i, std::memory_order_relaxed);
            }
        }

        bool push(T&& data) noexcept {
            auto pos = tail.fetch_add(1, std::memory_order_relaxed);
            auto& slot = slots[pos & (Capacity - 1)];
            while (slot.sequence.load(std::memory_order_acquire) != pos) {
            }
            slot.data = std::move(data);
            slot.sequence.store(pos + 1, std::memory_order_release);
            return true;
        }

        bool pop(T& data) noexcept {
            auto pos = head.load(std::memory_order_relaxed);
            auto& slot = slots[pos & (Capacity - 1)];
            if (slot.sequence.load(std::memory_order_acquire) != pos + 1) {
                return false;
            }
            data = std::move(slot.data);
            slot.sequence.store(pos + Capacity, std::memory_order_release);
            head.store(pos + 1, std::memory_order_relaxed);
            return true;
        }
    };

    // ──────────────────────────────────────────────────────────────────────────
    // Lock-Free SPMC Queue
    // ──────────────────────────────────────────────────────────────────────────
    template <typename T>
    class SPMCQueue {
        struct Node {
            T data;
            std::atomic<Node*> next{nullptr};

            explicit Node(T&& data) : data(std::move(data)) {
            }

            Node() = default;
        };

        std::atomic<Node*> head{nullptr};
        Node* tail{nullptr};

    public:
        SPMCQueue() = default;

        bool init() noexcept {
            auto dummy = new(std::nothrow) Node();
            if (!dummy) {
                return false;
            }
            head.store(dummy, std::memory_order_release);
            tail = dummy;
            return true;
        }

        bool push(T&& item) noexcept {
            auto* node = new(std::nothrow) Node(std::move(item));
            if (!node) {
                return false;
            }
            tail->next.store(node, std::memory_order_release);
            tail = node;
            return true;
        }

        bool pop(T& item) noexcept {
            auto* oldHead = head.load(std::memory_order_acquire);
            if (!oldHead) {
                return false;
            }
            while (true) {
                auto* next = oldHead->next.load(std::memory_order_acquire);
                if (!next) {
                    return false;
                }
                if (head.compare_exchange_weak(oldHead, next, std::memory_order_release, std::memory_order_acquire)) {
                    item = std::move(next->data);
                    delete oldHead;
                    return true;
                }
            }
        }

        ~SPMCQueue() {
            if (!head.load(std::memory_order_acquire)) {
                return;
            }
            T dummy;
            while (pop(dummy)) {
            }
            delete head.load(std::memory_order_acquire);
        }
    };

    // ──────────────────────────────────────────────────────────────────────────
    // Global Task Queue
    // ──────────────────────────────────────────────────────────────────────────

    inline MPSCQueue<TaskDescriptor, constants::QUEUE_CAPACITY>& getTaskQueue() noexcept {
        static MPSCQueue<TaskDescriptor, constants::QUEUE_CAPACITY> queue;
        return queue;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Global Error Queue
    // ──────────────────────────────────────────────────────────────────────────

    inline SPMCQueue<cuDAOStatus>& getErrorQueue() noexcept {
        static SPMCQueue<cuDAOStatus> queue;
        return queue;
    }
}