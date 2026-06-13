#pragma once
#include "05_task_descriptor.cuh"

namespace cuDAO {

    // ──────────────────────────────────────────────────────────────────────────
    // Kernel Parser
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * @brief Explicit read-only access annotation for a pointer argument.
     *
     * ReadWrapper is normally constructed with read(ptr). It tells cuDAO that
     * the submitted kernel reads from the pointer but does not write to it.
     */
    template <typename T>
    struct ReadWrapper {
        T* ptr; ///< Pointer passed to the CUDA kernel as a read dependency.

        explicit ReadWrapper(T* p) noexcept : ptr(p) {
        }
    };

    /**
     * @brief Explicit write access annotation for a pointer argument.
     *
     * WriteWrapper is normally constructed with write(ptr). It tells cuDAO that
     * the submitted kernel may write to the pointer.
     */
    template <typename T>
    struct WriteWrapper {
        T* ptr; ///< Pointer passed to the CUDA kernel as a write dependency.

        explicit WriteWrapper(T* p) noexcept : ptr(p) {
        }
    };

    /**
     * @brief Mark a pointer argument as read-only for dependency tracking.
     *
     * @param ptr Pointer that will be read by the submitted kernel.
     * @return A wrapper consumed by launchKernel() or launchKernelSync().
     */
    template <typename T>
    ReadWrapper<T> read(T* ptr) noexcept {
        return ReadWrapper<T>{ptr};
    }

    /**
     * @brief Mark a pointer argument as written for dependency tracking.
     *
     * @param ptr Pointer that will be written by the submitted kernel.
     * @return A wrapper consumed by launchKernel() or launchKernelSync().
     */
    template <typename T>
    WriteWrapper<T> write(T* ptr) noexcept {
        return WriteWrapper<T>{ptr};
    }

    template <typename T>
    struct is_read_wrapper : std::false_type {
    };

    template <typename T>
    struct is_read_wrapper<ReadWrapper<T>> : std::true_type {
    };

    template <typename T>
    struct is_write_wrapper : std::false_type {
    };

    template <typename T>
    struct is_write_wrapper<WriteWrapper<T>> : std::true_type {
    };

    template <typename T>
    struct is_cuda_ptr : std::false_type {
    };

    template <typename T>
    struct is_cuda_ptr<T*> : std::true_type {
    };

    template <typename T>
    struct is_cuda_ptr<const T*> : std::true_type {
    };

    template <typename T>
    void processArg(TaskDescriptor& desc, size_t& offset, T&& arg) {
        using Raw = std::decay_t<T>;
        if (desc.paramCount >= constants::MAX_PARAM_COUNT) {
            throw std::runtime_error("Too many parameters.");
        }
        if constexpr (is_write_wrapper<Raw>::value) {
            auto* ptr = static_cast<void*>(arg.ptr);
            std::memcpy(desc.paramBuffer.data() + offset, &ptr, sizeof(void*));
            desc.paramOffsets[desc.paramCount] = offset;
            desc.paramSizes[desc.paramCount] = sizeof(void*);
            desc.writeArgs[desc.writeArgsCount++] = ptr;
            desc.paramCount++;
            offset += sizeof(void*);
        }
        else if constexpr (is_read_wrapper<Raw>::value) {
            auto* ptr = static_cast<void*>(arg.ptr);
            std::memcpy(desc.paramBuffer.data() + offset, &ptr, sizeof(void*));
            desc.paramOffsets[desc.paramCount] = offset;
            desc.paramSizes[desc.paramCount] = sizeof(void*);
            desc.readArgs[desc.readArgsCount++] = ptr;
            desc.paramCount++;
            offset += sizeof(void*);
        }
        else if constexpr (std::is_pointer_v<Raw>) {
            if constexpr (std::is_const_v<std::remove_pointer_t<Raw>>) {
                // const T* -> void*
                auto* ptr = static_cast<void*>(const_cast<std::remove_const_t<std::remove_pointer_t<Raw>>*>(arg));
                std::memcpy(desc.paramBuffer.data() + offset, &ptr, sizeof(void*));
                desc.paramOffsets[desc.paramCount] = offset;
                desc.paramSizes[desc.paramCount] = sizeof(void*);
                desc.readArgs[desc.readArgsCount++] = ptr;
                desc.paramCount++;
                offset += sizeof(void*);
            }
            else {
                // T* -> void*
                auto* ptr = static_cast<void*>(arg);
                std::memcpy(desc.paramBuffer.data() + offset, &ptr, sizeof(void*));
                desc.paramOffsets[desc.paramCount] = offset;
                desc.paramSizes[desc.paramCount] = sizeof(void*);
                desc.writeArgs[desc.writeArgsCount++] = ptr;
                desc.paramCount++;
                offset += sizeof(void*);
            }
        }
        else {
            Raw val = arg;
            std::memcpy(desc.paramBuffer.data() + offset, &val, sizeof(Raw));
            desc.paramOffsets[desc.paramCount] = offset;
            desc.paramSizes[desc.paramCount] = sizeof(Raw);
            desc.paramCount++;
            offset += sizeof(Raw);
        }
    }

    template <typename Func, typename... Args>
    TaskDescriptor buildTask(Func func, const dim3 grid, const dim3 block, const size_t sharedMem, Args&&... args) {
        TaskDescriptor desc{};
        desc.func = reinterpret_cast<void*>(func);
        desc.grid = grid;
        desc.block = block;
        desc.sharedMem = sharedMem;

        size_t offset = 0;
        (processArg(desc, offset, std::forward<Args>(args)), ...);
        return desc;
    }
}