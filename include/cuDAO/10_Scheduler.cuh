#pragma once
#include "09_stream_pool.cuh"

namespace cuDAO {

    // ──────────────────────────────────────────────────────────────────────────
    // Scheduler
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * @brief Dedicated scheduler that converts cuDAO task descriptors into ordered CUDA stream work.
     *
     * Scheduler owns the CUDA streams, version slots, pointer slot map access,
     * scheduler thread, and CUDA function cache. Public APIs submit TaskDescriptor
     * objects to the global task queue; the scheduler thread pops and executes them.
     *
     * @tparam Policy Stream selection policy, usually RoundRobinPolicy or
     *                      LeastTaskPolicy.
     */
    template <typename Policy = RoundRobinPolicy>
    class Scheduler {
        using TaskQueueT = MPSCQueue<TaskDescriptor, constants::QUEUE_CAPACITY>;
        using ErrorQueueT = SPMCQueue<cuDAOStatus>;

        friend cuDAOStatus deviceSynchronize() noexcept;

        StreamPool<Policy> streamPool;
        VersionSlotPool slotPool{};
        SlotMapT* slotMap{nullptr};
        TaskQueueT* taskQueue{nullptr};
        ErrorQueueT* errorQueue{nullptr};
        WakeFlagT wakeFlag{0};
        std::atomic<bool> stopped{false};
        std::thread thread;
        CUdevice device;
        std::atomic<bool> initialized{false};
        std::unordered_map<void*, CUfunction> funcCache;
        std::array<VersionSlot*, constants::MAX_PARAM_COUNT> writeSlotsCache;
        void* kernelParams[constants::MAX_PARAM_COUNT];

        std::atomic<bool> idle{false};

        /**
         * @brief Retain and set the CUDA primary context for the scheduler device.
         *
         * @return Success on context setup, otherwise a CUDA driver error status.
         */
        cuDAOStatus initCudaContext() const noexcept {
            CUcontext ctx;
            auto re = cuDevicePrimaryCtxRetain(&ctx, device);
            if (re != CUDA_SUCCESS) {
                return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
            }
            re = cuCtxSetCurrent(ctx);
            if (re != CUDA_SUCCESS) {
                return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
            }
            return cuDAOStatus{cuDAOError::Success};
        }

        /**
         * @brief Initialize streams, version-slot backing memory, queues, and maps.
         *
         * @return Success if all scheduler resources were initialized.
         */
        cuDAOStatus initResource() noexcept {
            auto re = streamPool.init();
            if (re != CUDA_SUCCESS) {
                return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
            }
            re = slotPool.init();
            if (re != CUDA_SUCCESS) {
                return cuDAOStatus{cuDAOError::CudaDriverError, __func__, re};
            }
            try {
                slotMap = &getSlotMap();
                taskQueue = &getTaskQueue();
            }
            catch (const std::bad_alloc&) {
                return cuDAOStatus{cuDAOError::HostAllocationFailed, __func__};
            }
            errorQueue = &getErrorQueue();
            if (!errorQueue->init()) {
                return cuDAOStatus{cuDAOError::HostAllocationFailed, __func__};
            }
            return cuDAOStatus{cuDAOError::Success};
        }

        /**
         * @brief Release the CUDA primary context retained by initCudaContext().
         */
        void destroyCudaContext() const noexcept {
            cuDevicePrimaryCtxRelease(device);
        }

        /**
         * @brief Destroy scheduler-owned runtime resources.
         */
        void destroyResource() noexcept {
            streamPool.destroy();
            slotPool.destroy();
        }

        /**
         * @brief Resolve and cache a CUDA kernel function symbol.
         *
         * @param funcPtr Host-side CUDA kernel symbol pointer.
         * @param kernel Output CUDA Driver API function handle.
         * @return true if the function was resolved or found in the cache.
         */
        bool getCudaFunction(void* funcPtr, CUfunction& kernel) noexcept {
            if (const auto it = funcCache.find(funcPtr); it != funcCache.end()) {
                kernel = it->second;
                return true;
            }
            CUfunction func;
            if (const auto re = cudaGetFuncBySymbol(reinterpret_cast<cudaFunction_t*>(&func), funcPtr); re !=
                cudaSuccess) {
                return false;
            }
            funcCache[funcPtr] = func;
            kernel = func;
            return true;
        }

        /**
         * @brief Callback data used to mark read dependencies active before a task begins reading.
         */
        struct ReadCallbackData {
            std::array<VersionSlot*, constants::MAX_PARAM_COUNT>* readSlots;
            size_t readArgsCount;
            VersionSlotPool* slotPool;
        };

        /**
         * @brief Callback data used to release read dependencies and fulfill completion state.
         */
        struct CompletionCallbackData {
            std::array<VersionSlot*, constants::MAX_PARAM_COUNT> readSlots;
            size_t readArgsCount;
            VersionSlotPool* slotPool;
            CudaPromise* promise;
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            LeastTaskPolicy* policy;
            uint32_t streamId;
#endif
        };

        /**
         * @brief Callback data for pointer-specific sync tasks.
         */
        struct SyncCallbackData {
            CudaPromise* promise;
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            LeastTaskPolicy* policy;
            uint32_t streamId;
#endif
        };

        /**
         * @brief Callback data for scheduler-managed device free tasks.
         */
        struct FreeCallbackData {
            VersionSlotPool* slotPool;
            void* ptr;
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            LeastTaskPolicy* policy;
            uint32_t streamId;
#endif
        };

        /**
         * @brief Callback data for unregistering a pointer and optionally fulfilling a promise.
         */
        struct UnregisterCallbackData {
            VersionSlotPool* slotPool;
            void* ptr;
            CudaPromise* promise;
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            LeastTaskPolicy* policy;
            uint32_t streamId;
#endif
        };

        /**
         * @brief Stream callback that marks read slots as active.
         *
         * Increments each read slot's pending-read count and sets its read gate to one.
         * The callback is ordered before the CUDA operation that consumes the reads.
         */
        static void readStartCallback(void* data) noexcept {
            auto* data_ = reinterpret_cast<ReadCallbackData*>(data);
            auto* slotPool_ = data_->slotPool;
            auto readArgsCount = data_->readArgsCount;

            for (size_t i = 0; i < readArgsCount; ++i) {
                auto slot = data_->readSlots->at(i);
                ++slot->pendingReads;
                *slot->getReadGateAddr(slotPool_->pinnedMem) = 1;
            }

            delete data_;
        }

        /**
         * @brief Stream callback that fulfills a synchronization promise.
         *
         * Under the least-task policy, this callback also marks the selected stream task
         * as complete.
         */
        static void syncCallback(void* data) noexcept {
            auto* data_ = reinterpret_cast<SyncCallbackData*>(data);
            auto* promise = data_->promise;
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            auto* policy = data_->policy;
            auto streamId = data_->streamId;
            policy->complete(streamId);
#endif
            if (promise) {
                promise->set();
            }
            delete data_;
        }

        /**
         * @brief Stream callback that unregisters a pointer after asynchronous free.
         *
         * The associated slot is returned to the pool only after the stream reaches this
         * callback.
         */
        static void freeCallback(void* data) noexcept {
            auto* data_ = reinterpret_cast<FreeCallbackData*>(data);
            auto* ptr = data_->ptr;
            auto* slotPool = data_->slotPool;
            unregisterPtr(ptr, *slotPool);
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            auto* policy = data_->policy;
            auto streamId = data_->streamId;
            policy->complete(streamId);
#endif
            delete data_;
        }

        /**
         * @brief Stream callback that unregisters a pointer and fulfills the unregister promise.
         */
        static void unregisterCallback(void* data) noexcept {
            auto* data_ = reinterpret_cast<UnregisterCallbackData*>(data);
            auto* ptr = data_->ptr;
            auto* slotPool = data_->slotPool;
            auto* promise = data_->promise;
            unregisterPtr(ptr, *slotPool);
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            auto* policy = data_->policy;
            auto streamId = data_->streamId;
            policy->complete(streamId);
#endif
            promise->set();
            delete data_;
        }

        /**
         * @brief Stream callback that releases read gates and fulfills task completion state.
         *
         * Each read slot's pending-read count is decremented. If the count reaches zero,
         * the read gate is reset to zero so waiting writers may proceed.
         */
        static void completionCallBack(void* data) noexcept {
            auto* data_ = reinterpret_cast<CompletionCallbackData*>(data);
            auto* slotPool_ = data_->slotPool;
            auto* promise = data_->promise;
            auto readArgsCount = data_->readArgsCount;

            for (size_t i = 0; i < readArgsCount; ++i) {
                auto slot = data_->readSlots[i];
                if (--slot->pendingReads == 0) {
                    *slot->getReadGateAddr(slotPool_->pinnedMem) = 0;
                }
            }

            if (promise) {
                promise->set();
            }

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            data_->policy->complete(data_->streamId);
#endif

            delete data_;
        }

        /**
         * @brief Process a scheduler-managed kernel launch task.
         *
         * Registers read/write pointers, emits dependency waits, launches a read-start
         * callback, launches the CUDA kernel, publishes new write versions, and then
         * launches a completion callback.
         */
        void processKernelTask(TaskDescriptor& task) noexcept {
            // Phase 1 : Reversible operations
            // Allocate callback data
            auto* completionData = new(std::nothrow) CompletionCallbackData;
            if (!completionData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }
            completionData->readArgsCount = task.readArgsCount;
            completionData->promise = task.promise.get();
            completionData->slotPool = &slotPool;
            auto* readData = new(std::nothrow) ReadCallbackData{
                &completionData->readSlots,
                task.readArgsCount,
                &slotPool
            };
            if (!readData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                delete completionData;
                return;
            }

            // Register parameters
            for (size_t i = 0; i < task.writeArgsCount; ++i) {
                auto writeArg = task.writeArgs[i];
                auto [it, inserted] = slotMap->try_emplace(writeArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete readData;
                        delete completionData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                writeSlotsCache[i] = slot;
            }
            for (size_t i = 0; i < task.readArgsCount; ++i) {
                auto readArg = task.readArgs[i];
                auto [it, inserted] = slotMap->try_emplace(readArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete readData;
                        delete completionData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                completionData->readSlots[i] = slot;
            }

            // Get kernel
            CUfunction kernel;
            if (!getCudaFunction(task.func, kernel)) {
                delete readData;
                delete completionData;
                errorQueue->push(cuDAOStatus{cuDAOError::InvalidDeviceFunctionSymbol, __func__});
                return;
            }

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            completionData->streamId = streamId;
            completionData->policy = &streamPool.policy;
#endif

            // Wait write version
            for (size_t i = 0; i < task.writeArgsCount; ++i) {
                auto* slot = writeSlotsCache[i];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
                ++slot->expectedWriteVersion;
            }
            for (size_t i = 0; i < task.readArgsCount; ++i) {
                auto* slot = completionData->readSlots[i];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
            }

            // Wait read gate
            for (size_t i = 0; i < task.writeArgsCount; ++i) {
                auto* slot = writeSlotsCache[i];

                CUDAO_ASSERT(cuStreamWaitValue64(
                    stream, reinterpret_cast<CUdeviceptr>(slot->getReadGateAddr(slotPool.pinnedMem)), 0,
                    CU_STREAM_WAIT_VALUE_EQ));
            }

            // Launch read start callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, readStartCallback, readData));

            // Launch kernel
            for (size_t i = 0; i < task.paramCount; ++i) {
                kernelParams[i] = task.paramBuffer.data() + task.paramOffsets[i];
            }
            CUDAO_ASSERT(cuLaunchKernel(kernel,
                task.grid.x, task.grid.y, task.grid.z,
                task.block.x, task.block.y, task.block.z,
                task.sharedMem, stream,
                kernelParams, nullptr));

            // Update write version
            for (size_t i = 0; i < task.writeArgsCount; ++i) {
                auto* slot = writeSlotsCache[i];

                CUDAO_ASSERT(cuStreamWriteValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WRITE_VALUE_DEFAULT));
            }

            // Launch completion callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, completionCallBack, completionData));
        }

        /**
         * @brief Process a pointer-specific synchronization task.
         *
         * Waits for the pointer's current write version and fulfills the task promise
         * from a stream-ordered callback. If the pointer has not been tracked yet, a
         * slot is allocated lazily and the promise is fulfilled immediately.
         */
        void processSyncTask(TaskDescriptor& task) noexcept {
            auto ptr = task.writeArgs[0];
            auto it = slotMap->find(ptr);
            // Phase 1 : Reversible operations
            // Register pointer
            if (it == slotMap->end()) {
                auto* slot = slotPool.alloc();
                if (!slot) {
                    errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                    return;
                }
                slotMap->emplace(ptr, slot);
                task.promise->set();
                return;
            }
            auto slot = it->second;

            // Allocate callback data
            auto* syncData = new(std::nothrow) SyncCallbackData{
                task.promise.get()
            };
            if (!syncData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            syncData->streamId = streamId;
            syncData->policy = &streamPool.policy;
#endif

            // Wait write version
            CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));

            // Launch sync callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, syncCallback, reinterpret_cast<void*>(syncData)));
        }

        /**
         * @brief Process a scheduler-managed device free task.
         *
         * Waits for prior writes and active reads, submits cuMemFreeAsync, resets the
         * device-side slot version, and unregisters the pointer in a stream callback.
         */
        void processFreeTask(TaskDescriptor& task) noexcept {
            auto ptr = task.writeArgs[0];
            auto it = slotMap->find(ptr);
            if (it == slotMap->end()) {
                // Phase 2 : Irreversible operations
                // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
                uint32_t streamId;
                auto stream = streamPool.get(&streamId);
#else
                auto stream = streamPool.get();
#endif

                // Free data
                CUDAO_ASSERT(cuMemFreeAsync(reinterpret_cast<CUdeviceptr>(ptr), stream));
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
                streamPool.policy.complete(streamId);
#endif
                return;
            }

            auto slot = it->second;

            // Phase 1 : Reversible operations
            // Allocate free callback data
            auto* freeData = new(std::nothrow) FreeCallbackData{
                &slotPool,
                ptr
            };
            if (!freeData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }

            // Phase 2 : irreversible operations
            // Get Stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            freeData->streamId = streamId;
            freeData->policy = &streamPool.policy;
#endif

            // Wait write version & read gate
            CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
            CUDAO_ASSERT(cuStreamWaitValue64(
                stream, reinterpret_cast<CUdeviceptr>(slot->getReadGateAddr(slotPool.pinnedMem)), 0,
                CU_STREAM_WAIT_VALUE_EQ));

            // Free data
            CUDAO_ASSERT(cuMemFreeAsync(reinterpret_cast<CUdeviceptr>(ptr), stream));

            // Reset slot device-side version before making the slot reusable.
            CUDAO_ASSERT(cuStreamWriteValue64(
                stream,
                slot->getWriteVersionAddr(slotPool.deviceMem),
                0,
                CU_STREAM_WRITE_VALUE_DEFAULT
            ));

            // Launch free callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, freeCallback, reinterpret_cast<void*>(freeData)));
        }

        /**
         * @brief Process a device-to-device copy task.
         *
         * The source is treated as a read dependency and the destination as a write
         * dependency.
         */
        void processMemcpyDtoDTask(TaskDescriptor& task) noexcept {
            // Phase 1 : Reversible operations
            // Allocate callback data
            auto* completionData = new(std::nothrow) CompletionCallbackData;
            if (!completionData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }
            completionData->readArgsCount = task.readArgsCount;
            completionData->promise = task.promise.get();
            completionData->slotPool = &slotPool;
            auto* readData = new(std::nothrow) ReadCallbackData{
                &completionData->readSlots,
                task.readArgsCount,
                &slotPool
            };
            if (!readData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                delete completionData;
                return;
            }

            // Register parameters
            {
                auto writeArg = task.writeArgs[0];
                auto [it, inserted] = slotMap->try_emplace(writeArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete readData;
                        delete completionData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                writeSlotsCache[0] = slot;
            }
            {
                auto readArg = task.readArgs[0];
                auto [it, inserted] = slotMap->try_emplace(readArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete readData;
                        delete completionData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                completionData->readSlots[0] = slot;
            }

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            completionData->streamId = streamId;
            completionData->policy = &streamPool.policy;
#endif

            // Wait write version
            {
                auto* slot = writeSlotsCache[0];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
                ++slot->expectedWriteVersion;
            }
            {
                auto* slot = completionData->readSlots[0];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
            }

            // Wait read gate
            {
                auto* slot = writeSlotsCache[0];

                CUDAO_ASSERT(cuStreamWaitValue64(
                    stream, reinterpret_cast<CUdeviceptr>(slot->getReadGateAddr(slotPool.pinnedMem)), 0,
                    CU_STREAM_WAIT_VALUE_EQ));
            }

            // Launch read start callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, readStartCallback, readData));

            // Memcpy
            CUDAO_ASSERT(
                cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(task.writeArgs[0]),reinterpret_cast<CUdeviceptr>(task.
                    readArgs[0]),task.sharedMem,stream));

            // Update write version
            {
                auto* slot = writeSlotsCache[0];

                CUDAO_ASSERT(cuStreamWriteValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WRITE_VALUE_DEFAULT));
            }

            // Launch completion callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, completionCallBack, completionData));
        }

        /**
         * @brief Process a host-to-device copy task.
         *
         * The device destination is treated as a write dependency. The host source is
         * not tracked as a scheduler slot.
         */
        void processMemcpyHtoDTask(TaskDescriptor& task) noexcept {
            // Phase 1 : Reversible operations
            // Allocate callback data
            auto* syncData = new(std::nothrow) SyncCallbackData{
                task.promise.get()
            };
            if (!syncData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }

            // Register parameters
            {
                auto writeArg = task.writeArgs[0];
                auto [it, inserted] = slotMap->try_emplace(writeArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete syncData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                writeSlotsCache[0] = slot;
            }

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            syncData->streamId = streamId;
            syncData->policy = &streamPool.policy;
#endif

            // Wait write version
            {
                auto* slot = writeSlotsCache[0];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
                ++slot->expectedWriteVersion;
            }

            // Wait read gate
            {
                auto* slot = writeSlotsCache[0];

                CUDAO_ASSERT(cuStreamWaitValue64(
                    stream, reinterpret_cast<CUdeviceptr>(slot->getReadGateAddr(slotPool.pinnedMem)), 0,
                    CU_STREAM_WAIT_VALUE_EQ));
            }

            // Memcpy
            CUDAO_ASSERT(
                cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(task.writeArgs[0]),task.readArgs[0],task.sharedMem,
                    stream));

            // Update write version
            {
                auto* slot = writeSlotsCache[0];

                CUDAO_ASSERT(cuStreamWriteValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WRITE_VALUE_DEFAULT));
            }

            // Launch sync callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, syncCallback, reinterpret_cast<void*>(syncData)));
        }

        /**
         * @brief Process a device-to-host copy task.
         *
         * The device source is treated as a read dependency. The host destination is not
         * tracked as a scheduler slot; host completion should be observed with
         * cuDAOMemcpySync().
         */
        void processMemcpyDtoHTask(TaskDescriptor& task) noexcept {
            // Phase 1 : Reversible operations
            // Allocate callback data
            auto* completionData = new(std::nothrow) CompletionCallbackData;
            if (!completionData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }
            completionData->readArgsCount = task.readArgsCount;
            completionData->promise = task.promise.get();
            completionData->slotPool = &slotPool;
            auto* readData = new(std::nothrow) ReadCallbackData{
                &completionData->readSlots,
                task.readArgsCount,
                &slotPool
            };
            if (!readData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                delete completionData;
                return;
            }

            // Register parameters
            {
                auto readArg = task.readArgs[0];
                auto [it, inserted] = slotMap->try_emplace(readArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete readData;
                        delete completionData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                completionData->readSlots[0] = slot;
            }

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            completionData->streamId = streamId;
            completionData->policy = &streamPool.policy;
#endif

            // Wait write version
            {
                auto* slot = completionData->readSlots[0];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
            }

            // Launch read start callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, readStartCallback, readData));

            // Memcpy
            CUDAO_ASSERT(
                cuMemcpyDtoHAsync(task.writeArgs[0],reinterpret_cast<CUdeviceptr>(task.readArgs[0]),task.sharedMem,
                    stream));

            // Launch completion callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, completionCallBack, completionData));
        }

        /**
         * @brief Process a copy task where unified memory participates on both sides.
         *
         * The source is treated as a read dependency and the destination as a write
         * dependency.
         */
        void processMemcpyUtoUTask(TaskDescriptor& task) noexcept {
            // Phase 1 : Reversible operations
            // Allocate callback data
            auto* completionData = new(std::nothrow) CompletionCallbackData;
            if (!completionData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }
            completionData->readArgsCount = task.readArgsCount;
            completionData->promise = task.promise.get();
            completionData->slotPool = &slotPool;
            auto* readData = new(std::nothrow) ReadCallbackData{
                &completionData->readSlots,
                task.readArgsCount,
                &slotPool
            };
            if (!readData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                delete completionData;
                return;
            }

            // Register parameters
            {
                auto writeArg = task.writeArgs[0];
                auto [it, inserted] = slotMap->try_emplace(writeArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete readData;
                        delete completionData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                writeSlotsCache[0] = slot;
            }
            {
                auto readArg = task.readArgs[0];
                auto [it, inserted] = slotMap->try_emplace(readArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete readData;
                        delete completionData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                completionData->readSlots[0] = slot;
            }

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            completionData->streamId = streamId;
            completionData->policy = &streamPool.policy;
#endif

            // Wait write version
            {
                auto* slot = writeSlotsCache[0];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
                ++slot->expectedWriteVersion;
            }
            {
                auto* slot = completionData->readSlots[0];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
            }

            // Wait read gate
            {
                auto* slot = writeSlotsCache[0];

                CUDAO_ASSERT(cuStreamWaitValue64(
                    stream, reinterpret_cast<CUdeviceptr>(slot->getReadGateAddr(slotPool.pinnedMem)), 0,
                    CU_STREAM_WAIT_VALUE_EQ));
            }

            // Launch read start callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, readStartCallback, readData));

            // Memcpy
            CUDAO_ASSERT(
                cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(task.writeArgs[0]),reinterpret_cast<CUdeviceptr>(task.
                    readArgs[0]),task.sharedMem,stream));

            // Update write version
            {
                auto* slot = writeSlotsCache[0];

                CUDAO_ASSERT(cuStreamWriteValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WRITE_VALUE_DEFAULT));
            }

            // Launch completion callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, completionCallBack, completionData));
        }

        /**
         * @brief Process a host-to-unified-memory copy task.
         *
         * The unified destination is treated as a write dependency. The host source is
         * not tracked as a scheduler slot.
         */
        void processMemcpyHtoUTask(TaskDescriptor& task) noexcept {
            // Phase 1 : Reversible operations
            // Allocate callback data
            auto* syncData = new(std::nothrow) SyncCallbackData{
                task.promise.get()
            };
            if (!syncData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }

            // Register parameters
            {
                auto writeArg = task.writeArgs[0];
                auto [it, inserted] = slotMap->try_emplace(writeArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete syncData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                writeSlotsCache[0] = slot;
            }

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            syncData->streamId = streamId;
            syncData->policy = &streamPool.policy;
#endif

            // Wait write version
            {
                auto* slot = writeSlotsCache[0];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
                ++slot->expectedWriteVersion;
            }

            // Wait read gate
            {
                auto* slot = writeSlotsCache[0];

                CUDAO_ASSERT(cuStreamWaitValue64(
                    stream, reinterpret_cast<CUdeviceptr>(slot->getReadGateAddr(slotPool.pinnedMem)), 0,
                    CU_STREAM_WAIT_VALUE_EQ));
            }

            // Memcpy
            CUDAO_ASSERT(
                cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(task.writeArgs[0]),reinterpret_cast<CUdeviceptr>(task.
                    readArgs[0]),task.sharedMem,stream));

            // Update write version
            {
                auto* slot = writeSlotsCache[0];

                CUDAO_ASSERT(cuStreamWriteValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WRITE_VALUE_DEFAULT));
            }

            // Launch sync callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, syncCallback, reinterpret_cast<void*>(syncData)));
        }

        /**
         * @brief Process a unified-memory-to-host copy task.
         *
         * The unified source is treated as a read dependency. The host destination is
         * not tracked as a scheduler slot.
         */
        void processMemcpyUtoHTask(TaskDescriptor& task) noexcept {
            // Phase 1 : Reversible operations
            // Allocate callback data
            auto* completionData = new(std::nothrow) CompletionCallbackData;
            if (!completionData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }
            completionData->readArgsCount = task.readArgsCount;
            completionData->promise = task.promise.get();
            completionData->slotPool = &slotPool;
            auto* readData = new(std::nothrow) ReadCallbackData{
                &completionData->readSlots,
                task.readArgsCount,
                &slotPool
            };
            if (!readData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                delete completionData;
                return;
            }

            // Register parameters
            {
                auto readArg = task.readArgs[0];
                auto [it, inserted] = slotMap->try_emplace(readArg, nullptr);
                if (inserted) {
                    auto* slot = slotPool.alloc();
                    if (!slot) {
                        slotMap->erase(it);
                        delete readData;
                        delete completionData;
                        errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                        return;
                    }
                    it->second = slot;
                }
                auto slot = it->second;
                completionData->readSlots[0] = slot;
            }

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            completionData->streamId = streamId;
            completionData->policy = &streamPool.policy;
#endif

            // Wait write version
            {
                auto* slot = completionData->readSlots[0];
                CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                    slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
            }

            // Launch read start callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, readStartCallback, readData));

            // Memcpy
            CUDAO_ASSERT(
                cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(task.writeArgs[0]),reinterpret_cast<CUdeviceptr>(task.
                    readArgs[0]),task.sharedMem,stream));

            // Launch completion callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, completionCallBack, completionData));
        }

        /**
         * @brief Register a pointer with the scheduler without submitting CUDA work.
         *
         * Used by cuDAOMalloc() after a CUDA allocation has produced a pointer.
         */
        void processRegisterTask(TaskDescriptor& task) noexcept {
            // Register parameters
            auto readArg = task.readArgs[0];
            auto [it, inserted] = slotMap->try_emplace(readArg, nullptr);
            if (inserted) {
                auto* slot = slotPool.alloc();
                if (!slot) {
                    slotMap->erase(it);
                    errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                    return;
                }
                it->second = slot;
            }
        }

        /**
         * @brief Process a scheduler-managed asynchronous device allocation task.
         *
         * Submits cuMemAllocAsync, signals when the virtual address is available,
         * registers the resulting pointer, and publishes the initial write version.
         */
        void processAllocTask(TaskDescriptor& task) noexcept {
            // Phase 1
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            auto* syncData = new(std::nothrow) SyncCallbackData;
            if (!syncData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }
#endif

            // Phase 2
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            syncData->streamId = streamId;
            syncData->policy = &streamPool.policy;
#endif

            CUDAO_ASSERT(cuMemAllocAsync(reinterpret_cast<CUdeviceptr*>(task.writeArgs[0]), task.sharedMem, stream));

            task.promise->set();

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            CUDAO_ASSERT(cuLaunchHostFunc(stream, syncCallback, reinterpret_cast<void*>(syncData)));
#endif
            // Register
            auto* ptr = *reinterpret_cast<void**>(task.writeArgs[0]);
            auto [it, inserted] = slotMap->try_emplace(ptr, nullptr);
            if (inserted) {
                auto* slot = slotPool.alloc();
                if (!slot) {
                    slotMap->erase(it);
                    errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                    return;
                }
                it->second = slot;
            }
            auto* ptrSlot = it->second;

            ptrSlot->expectedWriteVersion = 1;
            CUDAO_ASSERT(cuStreamWriteValue64(stream, ptrSlot->getWriteVersionAddr(slotPool.deviceMem),
                ptrSlot->expectedWriteVersion, CU_STREAM_WRITE_VALUE_DEFAULT));
        }

        /**
         * @brief Unregister a pointer after all prior work on it is complete.
         *
         * Waits for the current write version and read gate, resets the device-side
         * write version, then removes the pointer-to-slot mapping in a stream callback.
         */
        void processUnregisterTask(TaskDescriptor& task) noexcept {
            auto ptr = task.writeArgs[0];
            auto it = slotMap->find(ptr);
            if (it == slotMap->end()) {
                task.promise->set();
                return;
            }

            auto slot = it->second;

            // Phase 1 : Reversible operations
            // Allocate free callback data
            auto* unregisterData = new(std::nothrow) UnregisterCallbackData{
                &slotPool,
                ptr,
                task.promise.get()
            };
            if (!unregisterData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }

            // Phase 2 : irreversible operations
            // Get Stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            unregisterData->streamId = streamId;
            unregisterData->policy = &streamPool.policy;
#endif

            // Wait write version & read gate
            CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
            CUDAO_ASSERT(cuStreamWaitValue64(
                stream, reinterpret_cast<CUdeviceptr>(slot->getReadGateAddr(slotPool.pinnedMem)), 0,
                CU_STREAM_WAIT_VALUE_EQ));

            // Reset write version
            CUDAO_ASSERT(cuStreamWriteValue64(
                stream,
                slot->getWriteVersionAddr(slotPool.deviceMem),
                0,
                CU_STREAM_WRITE_VALUE_DEFAULT
            ));

            // Launch callback
            CUDAO_ASSERT(cuLaunchHostFunc(stream, unregisterCallback, reinterpret_cast<void*>(unregisterData)));
        }

        /**
         * @brief Process a scheduler-managed memset task.
         *
         * Memset is modeled as a write dependency. The selected CUDA Driver API memset
         * primitive depends on the value width stored in the task descriptor.
         */
        void processMemsetTask(TaskDescriptor& task) noexcept {
            // Phase 1 : Reversible operations
            // Allocate callback data
            SyncCallbackData* syncData = nullptr;
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            syncData = new(std::nothrow) SyncCallbackData{
                task.promise.get()
            };
            if (!syncData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }
#endif
#ifndef CUDA_DAO_USE_LEAST_TASK_POLICY
            if (task.promise) {
                syncData = new(std::nothrow) SyncCallbackData{
                    task.promise.get()
                };
                if (!syncData) {
                    errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                    return;
                }
            }
#endif
            // Register parameters
            auto writeArg = task.writeArgs[0];
            auto [it, inserted] = slotMap->try_emplace(writeArg, nullptr);
            if (inserted) {
                auto* slot = slotPool.alloc();
                if (!slot) {
                    slotMap->erase(it);
                    if (syncData) {
                        delete syncData;
                    }
                    errorQueue->push(cuDAOStatus{cuDAOError::SlotPoolExhausted, __func__});
                    return;
                }
                it->second = slot;
            }
            auto slot = it->second;

            // Phase 2 : Irreversible operations
            // Get stream
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            uint32_t streamId;
            auto stream = streamPool.get(&streamId);
#else
            auto stream = streamPool.get();
#endif

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            syncData->streamId = streamId;
            syncData->policy = &streamPool.policy;
#endif

            // Wait write version
            CUDAO_ASSERT(cuStreamWaitValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                slot->expectedWriteVersion, CU_STREAM_WAIT_VALUE_GEQ));
            ++slot->expectedWriteVersion;

            // Wait read gate
            CUDAO_ASSERT(cuStreamWaitValue64(
                stream, reinterpret_cast<CUdeviceptr>(slot->getReadGateAddr(slotPool.pinnedMem)), 0,
                CU_STREAM_WAIT_VALUE_EQ));

            // Memset
            switch (task.paramSizes[0]) {
            case 1:
                {
                    uint8_t v;
                    std::memcpy(&v, task.paramBuffer.data(), 1);
                    CUDAO_ASSERT(
                        cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(task.writeArgs[0]), v, task.sharedMem, stream));
                    break;
                }
            case 2:
                {
                    uint16_t v;
                    std::memcpy(&v, task.paramBuffer.data(), 2);
                    CUDAO_ASSERT(
                        cuMemsetD16Async(reinterpret_cast<CUdeviceptr>(task.writeArgs[0]), v, task.sharedMem, stream));
                    break;
                }
            case 4:
                {
                    uint32_t v;
                    std::memcpy(&v, task.paramBuffer.data(), 4);
                    CUDAO_ASSERT(
                        cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(task.writeArgs[0]), v, task.sharedMem, stream));
                    break;
                }
            default:
                {
                    // Should never be reached
                    break;
                }
            }

            // Update write version
            CUDAO_ASSERT(cuStreamWriteValue64(stream, slot->getWriteVersionAddr(slotPool.deviceMem),
                slot->expectedWriteVersion, CU_STREAM_WRITE_VALUE_DEFAULT));

            // Launch callback
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            CUDAO_ASSERT(cuLaunchHostFunc(stream, syncCallback, reinterpret_cast<void*>(syncData)));
#endif
#ifndef CUDA_DAO_USE_LEAST_TASK_POLICY
            if (task.promise && syncData) {
                CUDAO_ASSERT(cuLaunchHostFunc(stream, syncCallback, reinterpret_cast<void*>(syncData)));
            }
#endif
        }

        /**
         * @brief Dispatch one task descriptor to the matching task-type handler.
         */
        void processTask(TaskDescriptor& task) noexcept {
            switch (task.taskType) {
            case TaskType::Kernel:
                processKernelTask(task);
                break;
            case TaskType::Sync:
                processSyncTask(task);
                break;
            case TaskType::Free:
                processFreeTask(task);
                break;
            case TaskType::MemcpyDtoD:
                processMemcpyDtoDTask(task);
                break;
            case TaskType::MemcpyDtoH:
                processMemcpyDtoHTask(task);
                break;
            case TaskType::MemcpyHtoD:
                processMemcpyHtoDTask(task);
                break;
            case TaskType::MemcpyHtoU:
                processMemcpyHtoUTask(task);
                break;
            case TaskType::MemcpyUtoH:
                processMemcpyUtoHTask(task);
                break;
            case TaskType::MemcpyUtoU:
                processMemcpyUtoUTask(task);
                break;
            case TaskType::Register:
                processRegisterTask(task);
                break;
            case TaskType::Alloc:
                processAllocTask(task);
                break;
            case TaskType::Unregister:
                processUnregisterTask(task);
                break;
            case TaskType::Memset:
                processMemsetTask(task);
                break;
            default:
                // Should never be reached
                break;
            }
        }

        /**
         * @brief Main scheduler thread loop.
         *
         * Initializes resources, consumes tasks until stop is requested, performs a
         * short spin before sleeping, and wakes through the platform wake flag when new
         * tasks are submitted.
         */
        void run() {
            auto re = initCudaContext();
            if (re.err != cuDAOError::Success) {
                initStatus = cuDAOStatus{re};
                initialized.store(true, std::memory_order_release);
                return;
            }
            re = initResource();
            if (re.err != cuDAOError::Success) {
                initStatus = cuDAOStatus{re};
                initialized.store(true, std::memory_order_release);
                return;
            }

            initialized.store(true, std::memory_order_release);

            while (!stopped.load(std::memory_order_relaxed)) {
                idle.store(false, std::memory_order_relaxed);
                TaskDescriptor task;

                while (taskQueue->pop(task)) {
                    processTask(task);
                }

                bool found = false;
                for (auto i = 0; i < constants::SCHEDULER_SPIN_COUNT && !found; ++i) {
                    if (taskQueue->pop(task)) {
                        processTask(task);
                        found = true;
                    }
                }
                if (found) {
                    idle.store(true, std::memory_order_release);
                    continue;
                }

                if (taskQueue->pop(task)) {
                    processTask(task);
                    idle.store(true, std::memory_order_release);
                    continue;
                }

                idle.store(true, std::memory_order_release);
                platformWait(wakeFlag);
            }

            destroyResource();
            destroyCudaContext();
        }

    public:
        cuDAOStatus initStatus{cuDAOError::Success};

        /**
         * @brief Start the scheduler thread for a CUDA device.
         *
         * @param device_ CUDA device ordinal used by the scheduler.
         */
        explicit Scheduler(const CUdevice device_) : device(device_) {
            thread = std::thread(&Scheduler::run, this);
            while (!initialized.load(std::memory_order_acquire)) {
            }
        }

        /**
         * @brief Stop the scheduler thread and release resources.
         */
        ~Scheduler() {
            stopped.store(true);
            platformNotify(wakeFlag);
            if (thread.joinable()) {
                thread.join();
            }
        }

        /**
         * @brief Submit a task descriptor and wake the scheduler thread.
         *
         * @param task Task descriptor to move into the global task queue.
         */
        void submitTask(TaskDescriptor&& task) noexcept {
            taskQueue->push(std::move(task));
            platformNotify(wakeFlag);
        }
    };

#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
    using DefaultScheduler = Scheduler<LeastTaskPolicy>;
#else
    using DefaultScheduler = Scheduler<RoundRobinPolicy>;
#endif

    /**
     * @brief Return the process-local default scheduler singleton.
     */
    inline DefaultScheduler& getDefaultScheduler() {
        static DefaultScheduler scheduler(0);
        return scheduler;
    }
}