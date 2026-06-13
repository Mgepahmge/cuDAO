#pragma once
#include "09_stream_pool.cuh"

namespace cuDAO {

    // ──────────────────────────────────────────────────────────────────────────
    // Scheduler
    // ──────────────────────────────────────────────────────────────────────────

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

        void destroyCudaContext() const noexcept {
            cuDevicePrimaryCtxRelease(device);
        }

        void destroyResource() noexcept {
            streamPool.destroy();
            slotPool.destroy();
        }

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

        struct ReadCallbackData {
            std::array<VersionSlot*, constants::MAX_PARAM_COUNT>* readSlots;
            size_t readArgsCount;
            VersionSlotPool* slotPool;
        };

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

        struct SyncCallbackData {
            CudaPromise* promise;
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            LeastTaskPolicy* policy;
            uint32_t streamId;
#endif
        };

        struct FreeCallbackData {
            VersionSlotPool* slotPool;
            void* ptr;
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            LeastTaskPolicy* policy;
            uint32_t streamId;
#endif
        };

        struct UnregisterCallbackData {
            VersionSlotPool* slotPool;
            void* ptr;
            CudaPromise* promise;
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            LeastTaskPolicy* policy;
            uint32_t streamId;
#endif
        };

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

        void processMemsetTask(TaskDescriptor& task) noexcept {
            // Phase 1 : Reversible operations
            // Allocate callback data
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
            auto* syncData = new(std::nothrow) SyncCallbackData{
                task.promise.get()
            };
            if (!syncData) {
                errorQueue->push(cuDAOStatus{cuDAOError::HostAllocationFailed, __func__});
                return;
            }
#endif
            // Register parameters
            auto writeArg = task.writeArgs[0];
            auto [it, inserted] = slotMap->try_emplace(writeArg, nullptr);
            if (inserted) {
                auto* slot = slotPool.alloc();
                if (!slot) {
                    slotMap->erase(it);
#ifdef CUDA_DAO_USE_LEAST_TASK_POLICY
                    delete syncData;
#endif
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
        }

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

        explicit Scheduler(const CUdevice device_) : device(device_) {
            thread = std::thread(&Scheduler::run, this);
            while (!initialized.load(std::memory_order_acquire)) {
            }
        }

        ~Scheduler() {
            stopped.store(true);
            platformNotify(wakeFlag);
            if (thread.joinable()) {
                thread.join();
            }
        }

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

    inline DefaultScheduler& getDefaultScheduler() {
        static DefaultScheduler scheduler(0);
        return scheduler;
    }
}