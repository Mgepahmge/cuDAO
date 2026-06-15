#pragma once
#include "06_kernel_parser.cuh"

namespace cuDAO {

    // ──────────────────────────────────────────────────────────────────────────
    // Wait-Free MPSC Queue
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * @brief Bounded multi-producer/single-consumer queue used for scheduler tasks.
     *
     * MPSCQueue is the submission path from public API threads to the dedicated
     * scheduler thread. Multiple producer threads may push concurrently, while only
     * the scheduler thread pops. The queue is a fixed-size ring buffer using
     * per-slot sequence numbers.
     *
     * @tparam T Stored item type.
     * @tparam Capacity Ring-buffer capacity. Must be a power of two.
     *
     * @note push() is wait-free with respect to allocation, but it may spin while
     *       waiting for a full slot to become available.
     */
    template <typename T, size_t Capacity>
    class MPSCQueue {
        static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");

        /**
         * @brief Single ring-buffer cell used by MPSCQueue.
         *
         * The sequence value indicates whether the slot is ready for a producer or for
         * the single consumer. It is the core state used to avoid a separate lock.
         */
        struct Slot {
            std::atomic<size_t> sequence;
            T data;
        };

        alignas(64) std::array<Slot, Capacity> slots;
        alignas(64) std::atomic<size_t> head;
        alignas(64) std::atomic<size_t> tail;

    public:
        /**
         * @brief Initialize all slot sequence numbers and reset head/tail positions.
         */
        MPSCQueue() noexcept : head(0), tail(0) {
            for (size_t i = 0; i < Capacity; ++i) {
                slots[i].sequence.store(i, std::memory_order_relaxed);
            }
        }

        /**
         * @brief Push one item into the queue.
         *
         * @param data Item to move into the queue.
         * @return Always true after the item has been stored.
         *
         * @note This function spins if the selected ring-buffer slot has not yet been
         *       released by the consumer.
         */
        bool push(T&& data) noexcept {
            auto pos = tail.fetch_add(1, std::memory_order_relaxed);
            auto& slot = slots[pos & (Capacity - 1)];
            while (slot.sequence.load(std::memory_order_acquire) != pos) {
            }
            slot.data = std::move(data);
            slot.sequence.store(pos + 1, std::memory_order_release);
            return true;
        }

        /**
         * @brief Try to pop one item from the queue.
         *
         * @param data Output object receiving the popped item.
         * @return true if an item was popped; false if the queue is currently empty.
         */
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
    /**
     * @brief Single-producer/multi-consumer queue used for asynchronous errors.
     *
     * The scheduler is the producer. Host code may poll errors through
     * cuDAOGetLastError(), which consumes items from this queue.
     *
     * @tparam T Stored item type.
     */
    template <typename T>
    class SPMCQueue {
        /**
         * @brief Linked-list node used by SPMCQueue.
         */
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

        /**
         * @brief Initialize the queue with a dummy node.
         *
         * @return true if the dummy node was allocated successfully.
         */
        bool init() noexcept {
            auto dummy = new(std::nothrow) Node();
            if (!dummy) {
                return false;
            }
            head.store(dummy, std::memory_order_release);
            tail = dummy;
            return true;
        }

        /**
         * @brief Push one item into the error queue.
         *
         * @param item Item to move into the queue.
         * @return false if host allocation for the new node failed.
         */
        bool push(T&& item) noexcept {
            auto* node = new(std::nothrow) Node(std::move(item));
            if (!node) {
                return false;
            }
            tail->next.store(node, std::memory_order_release);
            tail = node;
            return true;
        }

        /**
         * @brief Try to pop one item from the error queue.
         *
         * @param item Output object receiving the popped item.
         * @return true if an item was popped; false if the queue is empty.
         */
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

        /**
         * @brief Destroy remaining linked-list nodes.
         */
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

    /**
     * @brief Return the global scheduler task queue.
     *
     * The queue is a function-local singleton shared by public API submission paths
     * and the default scheduler.
     */
    inline MPSCQueue<TaskDescriptor, constants::QUEUE_CAPACITY>& getTaskQueue() noexcept {
        static MPSCQueue<TaskDescriptor, constants::QUEUE_CAPACITY> queue;
        return queue;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Global Error Queue
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * @brief Return the global asynchronous error queue.
     *
     * Errors produced on the scheduler thread are pushed here and later retrieved
     * by cuDAOGetLastError().
     */
    inline SPMCQueue<cuDAOStatus>& getErrorQueue() noexcept {
        static SPMCQueue<cuDAOStatus> queue;
        return queue;
    }
}