#pragma once
#include "08_version_slot.cuh"

namespace cuDAO {

    // ──────────────────────────────────────────────────────────────────────────
    // Stream Pool
    // ──────────────────────────────────────────────────────────────────────────

    struct RoundRobinPolicy {
        uint32_t counter = 0;

        uint32_t select(const uint32_t streamCount) noexcept {
            return counter++ & (streamCount - 1);
        }
    };

    struct LeastTaskPolicy {
        std::array<std::atomic<uint32_t>, constants::STREAM_COUNT> taskCount{};

        uint32_t select(uint32_t streamCount) noexcept {
            uint32_t minIdx = 0;
            for (uint32_t i = 1; i < streamCount; ++i) {
                if (taskCount[i].load(std::memory_order_relaxed) < taskCount[minIdx].load(std::memory_order_relaxed)) {
                    minIdx = i;
                }
            }
            taskCount[minIdx].fetch_add(1, std::memory_order_relaxed);
            return minIdx;
        }

        void complete(uint32_t streamIdx) noexcept {
            taskCount[streamIdx].fetch_sub(1, std::memory_order_relaxed);
        }
    };

    template <typename Policy = RoundRobinPolicy>
    struct StreamPool {
        std::array<CUstream, constants::STREAM_COUNT> streams{};
        Policy policy;

        CUresult init() noexcept {
            for (uint32_t i = 0; i < constants::STREAM_COUNT; ++i) {
                CUresult res = cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING);
                if (res != CUDA_SUCCESS) {
                    for (uint32_t j = 0; j < i; ++j)
                        cuStreamDestroy(streams[j]);
                    return res;
                }
            }
            return CUDA_SUCCESS;
        }

        void destroy() noexcept {
            for (auto& stream : streams)
                cuStreamDestroy(stream);
        }

        CUstream get(uint32_t* outIdx = nullptr) noexcept {
            uint32_t idx = policy.select(constants::STREAM_COUNT);
            if (outIdx) *outIdx = idx;
            return streams[idx];
        }

        cuDAOStatus synchronizeAll() noexcept {
            std::unique_ptr<std::vector<std::pair<uint32_t, CUresult>>> status(nullptr);
            for (uint32_t i = 0; i < constants::STREAM_COUNT; ++i) {
                auto res = cuStreamSynchronize(streams[i]);
                if (res != CUDA_SUCCESS) {
                    if (!status) {
                        status = std::make_unique<std::vector<std::pair<uint32_t, CUresult>>>();
                    }
                    status->emplace_back(i, res);
                }
            }
            if (!status) {
                return cuDAOStatus{
                    cuDAOError::Success
                };
            }
            cuDAOStatus result{
                cuDAOError::SynchronizeFailed
            };
            std::string msgString{"Failed to synchronize the following streams:\n"};
            for (auto i = status->begin(); i != status->end(); ++i) {
                msgString += "Stream " + std::to_string(i->first) + " : ";
                const char* cudaErrStr;
                cuGetErrorString(i->second, &cudaErrStr);
                msgString += cudaErrStr;
                msgString += "\n";
            }
            result.msg = LazyString(msgString);
            return result;
        }
    };
}