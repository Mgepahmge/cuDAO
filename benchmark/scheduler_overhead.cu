#include <cuDAO.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string_view>
#include <string>
#include <array>
#include <utility>
#include <vector>

namespace {
    using Clock = std::chrono::steady_clock;

    constexpr dim3 kGrid{1, 1, 1};
    constexpr dim3 kBlock{1, 1, 1};

    struct BenchmarkConfig {
        size_t warmup{32};
        size_t trials{30};
        size_t batch{128};
        size_t copyBytes{4096};
    };

    struct SampleStats {
        double mean{};
        double min{};
        double median{};
        double max{};
    };

    struct BenchmarkResult {
        std::string name;
        std::vector<double> submitUsPerOp;
        std::vector<double> totalUsPerOp;
    };

    struct BenchmarkStats {
        SampleStats submit;
        SampleStats total;
    };

    [[noreturn]] void die(const char* message) {
        std::fputs(message, stderr);
        std::fputc('\n', stderr);
        std::exit(EXIT_FAILURE);
    }

    [[noreturn]] void dieCuda(const char* what, const cudaError_t err) {
        std::fprintf(stderr, "%s failed: %s (%d)\n", what, cudaGetErrorString(err), static_cast<int>(err));
        std::exit(EXIT_FAILURE);
    }

    [[noreturn]] void dieDriver(const char* what, const CUresult err) {
        const char* msg = nullptr;
        cuGetErrorString(err, &msg);
        std::fprintf(stderr, "%s failed: %s (%d)\n", what, msg ? msg : "unknown", static_cast<int>(err));
        std::exit(EXIT_FAILURE);
    }

    void dieStatus(const char* what, const cuDAO::cuDAOStatus& status) {
        if (status.err == cuDAO::cuDAOError::Success) {
            return;
        }
        std::fprintf(stderr, "%s failed: err=%d cudaResult=%d where=%s msg=%s\n",
                     what,
                     static_cast<int>(status.err),
                     static_cast<int>(status.cudaResult),
                     status.where ? status.where : "(null)",
                     status.msg.c_str());
        std::exit(EXIT_FAILURE);
    }

    bool parseSizeT(std::string_view value, size_t& out) {
        const auto* begin = value.data();
        const auto* end = value.data() + value.size();
        auto [ptr, ec] = std::from_chars(begin, end, out);
        return ec == std::errc{} && ptr == end;
    }

    BenchmarkConfig parseArgs(int argc, char** argv) {
        BenchmarkConfig cfg;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg{argv[i]};
            auto takeValue = [&](size_t& target) {
                if (i + 1 >= argc) {
                    die("missing value for benchmark option");
                }
                if (!parseSizeT(argv[++i], target)) {
                    die("invalid numeric benchmark option");
                }
            };

            if (arg == "--warmup") {
                takeValue(cfg.warmup);
            } else if (arg == "--trials") {
                takeValue(cfg.trials);
            } else if (arg == "--batch") {
                takeValue(cfg.batch);
            } else if (arg == "--copy-bytes") {
                takeValue(cfg.copyBytes);
            } else if (arg == "--help" || arg == "-h") {
                std::puts("Usage: benchmark_scheduler_overhead [--warmup N] [--trials N] [--batch N] [--copy-bytes N]");
                std::exit(EXIT_SUCCESS);
            } else {
                die("unknown benchmark option");
            }
        }
        if (cfg.batch == 0 || cfg.trials == 0 || cfg.copyBytes == 0) {
            die("benchmark batch, trials, and copy-bytes must be greater than zero");
        }
        return cfg;
    }

    SampleStats summarize(std::vector<double> samples) {
        if (samples.empty()) {
            return {};
        }
        std::sort(samples.begin(), samples.end());
        const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
        const size_t mid = samples.size() / 2;
        return SampleStats{
            sum / static_cast<double>(samples.size()),
            samples.front(),
            samples.size() % 2 == 0
                ? (samples[mid - 1] + samples[mid]) * 0.5
                : samples[mid],
            samples.back()
        };
    }

    std::string formatFixed(const double value) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%.3f", value);
        return buffer;
    }

    BenchmarkStats summarize(const BenchmarkResult& result) {
        return BenchmarkStats{
            summarize(result.submitUsPerOp),
            summarize(result.totalUsPerOp)
        };
    }

    std::vector<BenchmarkStats> summarizeAll(const std::vector<BenchmarkResult>& results) {
        std::vector<BenchmarkStats> stats;
        stats.reserve(results.size());
        for (const auto& result : results) {
            stats.push_back(summarize(result));
        }
        return stats;
    }

    void printTable(const std::vector<BenchmarkResult>& results) {
        const auto stats = summarizeAll(results);

        constexpr std::array<const char*, 8> headers{
            "submit mean",
            "submit min",
            "submit median",
            "submit max",
            "total mean",
            "total min",
            "total median",
            "total max"
        };
        std::array<size_t, headers.size()> widths{};
        size_t nameWidth = std::string{"benchmark"}.size();

        for (size_t i = 0; i < results.size(); ++i) {
            nameWidth = std::max(nameWidth, results[i].name.size());

            const auto updateWidth = [&](const size_t index, const SampleStats& s) {
                widths[index] = std::max(widths[index], std::string{headers[index]}.size());
                widths[index] = std::max(widths[index], formatFixed(s.mean).size());
                widths[index] = std::max(widths[index], formatFixed(s.min).size());
                widths[index] = std::max(widths[index], formatFixed(s.median).size());
                widths[index] = std::max(widths[index], formatFixed(s.max).size());
            };
            updateWidth(0, stats[i].submit);
            updateWidth(1, stats[i].submit);
            updateWidth(2, stats[i].submit);
            updateWidth(3, stats[i].submit);
            updateWidth(4, stats[i].total);
            updateWidth(5, stats[i].total);
            updateWidth(6, stats[i].total);
            updateWidth(7, stats[i].total);
        }

        const auto printCell = [](const std::string& text, const size_t width) {
            std::printf("%-*s", static_cast<int>(width), text.c_str());
        };

        std::printf("\n");
        printCell("benchmark", nameWidth);
        std::printf("  ");
        printCell(headers[0], widths[0]);
        std::printf("  ");
        printCell(headers[1], widths[1]);
        std::printf("  ");
        printCell(headers[2], widths[2]);
        std::printf("  ");
        printCell(headers[3], widths[3]);
        std::printf("  ");
        printCell(headers[4], widths[4]);
        std::printf("  ");
        printCell(headers[5], widths[5]);
        std::printf("  ");
        printCell(headers[6], widths[6]);
        std::printf("  ");
        printCell(headers[7], widths[7]);
        std::printf("\n");

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& row = results[i];
            const auto& s = stats[i];
            printCell(row.name, nameWidth);
            std::printf("  ");
            printCell(formatFixed(s.submit.mean), widths[0]);
            std::printf("  ");
            printCell(formatFixed(s.submit.min), widths[1]);
            std::printf("  ");
            printCell(formatFixed(s.submit.median), widths[2]);
            std::printf("  ");
            printCell(formatFixed(s.submit.max), widths[3]);
            std::printf("  ");
            printCell(formatFixed(s.total.mean), widths[4]);
            std::printf("  ");
            printCell(formatFixed(s.total.min), widths[5]);
            std::printf("  ");
            printCell(formatFixed(s.total.median), widths[6]);
            std::printf("  ");
            printCell(formatFixed(s.total.max), widths[7]);
            std::printf("\n");
        }
    }

    template <typename SubmitFn, typename DrainFn>
    BenchmarkResult runBenchmark(std::string name, const BenchmarkConfig& cfg, SubmitFn&& submit, DrainFn&& drain) {
        BenchmarkResult result;
        result.name = std::move(name);
        result.submitUsPerOp.reserve(cfg.trials);
        result.totalUsPerOp.reserve(cfg.trials);

        for (size_t i = 0; i < cfg.warmup; ++i) {
            for (size_t j = 0; j < cfg.batch; ++j) {
                submit(j);
            }
            drain();
        }

        for (size_t trial = 0; trial < cfg.trials; ++trial) {
            const auto totalStart = Clock::now();
            const auto submitStart = totalStart;
            for (size_t j = 0; j < cfg.batch; ++j) {
                submit(j);
            }
            const auto submitEnd = Clock::now();
            drain();
            const auto totalEnd = Clock::now();

            const auto submitElapsed = std::chrono::duration<double, std::micro>(submitEnd - submitStart).count();
            const auto totalElapsed = std::chrono::duration<double, std::micro>(totalEnd - totalStart).count();
            result.submitUsPerOp.push_back(submitElapsed / static_cast<double>(cfg.batch));
            result.totalUsPerOp.push_back(totalElapsed / static_cast<double>(cfg.batch));
        }

        return result;
    }

    __global__ void noopKernel() {
        asm volatile("");
    }

    __global__ void dependencyKernel(int* out, const int* in) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            out[0] = in[0] + 1;
        }
    }

    void* toVoidPtr(const CUdeviceptr ptr) {
        return reinterpret_cast<void*>(ptr);
    }

    struct DeviceBuffers {
        CUdeviceptr noopScratch{};
        CUdeviceptr depIn{};
        CUdeviceptr depOut{};
        CUdeviceptr memsetDst{};
        CUdeviceptr memcpySrc{};
        CUdeviceptr memcpyDst{};
        cudaStream_t baselineStream{};
    };

    DeviceBuffers createBuffers(const BenchmarkConfig& cfg) {
        DeviceBuffers buffers{};
        const size_t memsetBytes = std::max<size_t>(cfg.copyBytes, sizeof(uint32_t));
        CUresult re = cuMemAlloc(&buffers.noopScratch, sizeof(int));
        if (re != CUDA_SUCCESS) {
            dieDriver("cuMemAlloc(noopScratch)", re);
        }
        re = cuMemAlloc(&buffers.depIn, sizeof(int));
        if (re != CUDA_SUCCESS) {
            dieDriver("cuMemAlloc(depIn)", re);
        }
        re = cuMemAlloc(&buffers.depOut, sizeof(int));
        if (re != CUDA_SUCCESS) {
            dieDriver("cuMemAlloc(depOut)", re);
        }
        re = cuMemAlloc(&buffers.memsetDst, memsetBytes);
        if (re != CUDA_SUCCESS) {
            dieDriver("cuMemAlloc(memsetDst)", re);
        }
        re = cuMemAlloc(&buffers.memcpySrc, cfg.copyBytes);
        if (re != CUDA_SUCCESS) {
            dieDriver("cuMemAlloc(memcpySrc)", re);
        }
        re = cuMemAlloc(&buffers.memcpyDst, cfg.copyBytes);
        if (re != CUDA_SUCCESS) {
            dieDriver("cuMemAlloc(memcpyDst)", re);
        }

        const cudaError_t streamErr = cudaStreamCreateWithFlags(&buffers.baselineStream, cudaStreamNonBlocking);
        if (streamErr != cudaSuccess) {
            dieCuda("cudaStreamCreateWithFlags", streamErr);
        }

        const int zero = 0;
        re = cuMemcpyHtoD(buffers.depIn, &zero, sizeof(zero));
        if (re != CUDA_SUCCESS) {
            dieDriver("cuMemcpyHtoD(depIn)", re);
        }
        re = cuMemcpyHtoD(buffers.noopScratch, &zero, sizeof(zero));
        if (re != CUDA_SUCCESS) {
            dieDriver("cuMemcpyHtoD(noopScratch)", re);
        }
        re = cuMemcpyHtoD(buffers.memcpySrc, &zero, sizeof(zero));
        if (re != CUDA_SUCCESS) {
            dieDriver("cuMemcpyHtoD(memcpySrc)", re);
        }
        return buffers;
    }

    void destroyBuffers(DeviceBuffers& buffers) {
        cudaStreamDestroy(buffers.baselineStream);
        cuMemFree(buffers.memcpyDst);
        cuMemFree(buffers.memcpySrc);
        cuMemFree(buffers.memsetDst);
        cuMemFree(buffers.depOut);
        cuMemFree(buffers.depIn);
        cuMemFree(buffers.noopScratch);
    }

    BenchmarkResult runRawCudaNoop(const BenchmarkConfig& cfg, const DeviceBuffers& buffers) {
        return runBenchmark(
            "raw cuda noop",
            cfg,
            [&](size_t) {
                const cudaError_t err = cudaLaunchKernel(reinterpret_cast<const void*>(&noopKernel), kGrid, kBlock, nullptr, 0, buffers.baselineStream);
                if (err != cudaSuccess) {
                    dieCuda("cudaLaunchKernel", err);
                }
            },
            [&]() {
                const cudaError_t err = cudaStreamSynchronize(buffers.baselineStream);
                if (err != cudaSuccess) {
                    dieCuda("cudaStreamSynchronize", err);
                }
            }
        );
    }

    BenchmarkResult runCuDaoNoop(const BenchmarkConfig& cfg) {
        return runBenchmark(
            "cuDAO noop kernel",
            cfg,
            [&](size_t) {
                dieStatus("launchKernel(noopKernel)",
                          cuDAO::launchKernel(noopKernel, kGrid, kBlock, 0));
            },
            [&]() {
                dieStatus("deviceSynchronize", cuDAO::deviceSynchronize());
            }
        );
    }

    BenchmarkResult runCuDaoDependencyKernel(const BenchmarkConfig& cfg, const DeviceBuffers& buffers) {
        return runBenchmark(
            "cuDAO dep kernel",
            cfg,
            [&](size_t) {
                dieStatus("launchKernel(dependencyKernel)",
                          cuDAO::launchKernel(
                              dependencyKernel,
                              kGrid,
                              kBlock,
                              0,
                              cuDAO::write(reinterpret_cast<int*>(toVoidPtr(buffers.depOut))),
                              cuDAO::read(reinterpret_cast<int*>(toVoidPtr(buffers.depIn)))
                          ));
            },
            [&]() {
                dieStatus("deviceSynchronize", cuDAO::deviceSynchronize());
            }
        );
    }

    BenchmarkResult runCuDaoMemset(const BenchmarkConfig& cfg, const DeviceBuffers& buffers) {
        const size_t memsetCount = std::max<size_t>(1, cfg.copyBytes / sizeof(uint32_t));
        return runBenchmark(
            "cuDAO memset",
            cfg,
            [&](size_t) {
                dieStatus("cuDAOMemset",
                          cuDAO::cuDAOMemset(reinterpret_cast<uint32_t*>(toVoidPtr(buffers.memsetDst)),
                                             uint32_t{0x12345678u},
                                             memsetCount));
            },
            [&]() {
                dieStatus("deviceSynchronize", cuDAO::deviceSynchronize());
            }
        );
    }

    BenchmarkResult runCuDaoMemcpy(const BenchmarkConfig& cfg, const DeviceBuffers& buffers) {
        return runBenchmark(
            "cuDAO memcpy D2D",
            cfg,
            [&](size_t) {
                dieStatus("cuDAOMemcpy",
                          cuDAO::cuDAOMemcpy(
                              reinterpret_cast<int*>(toVoidPtr(buffers.memcpyDst)),
                              reinterpret_cast<const int*>(toVoidPtr(buffers.memcpySrc)),
                              cfg.copyBytes,
                              cuDAO::cuDAOMemcpyType::DeviceToDevice
                          ));
            },
            [&]() {
                dieStatus("deviceSynchronize", cuDAO::deviceSynchronize());
            }
        );
    }
}

int main(int argc, char** argv) {
    const auto cfg = parseArgs(argc, argv);

    int deviceCount = 0;
    const cudaError_t countErr = cudaGetDeviceCount(&deviceCount);
    if (countErr != cudaSuccess) {
        dieCuda("cudaGetDeviceCount", countErr);
    }
    if (deviceCount == 0) {
        die("no CUDA device found");
    }
    const cudaError_t setErr = cudaSetDevice(0);
    if (setErr != cudaSuccess) {
        dieCuda("cudaSetDevice", setErr);
    }

    dieStatus("cuDAOInit", cuDAO::cuDAOInit());

    DeviceBuffers buffers = createBuffers(cfg);

    std::puts("cuDAO scheduler overhead benchmark");
    std::printf("batch=%zu trials=%zu warmup=%zu copyBytes=%zu\n\n",
                cfg.batch, cfg.trials, cfg.warmup, cfg.copyBytes);

    std::vector<BenchmarkResult> results;
    results.push_back(runRawCudaNoop(cfg, buffers));
    results.push_back(runCuDaoNoop(cfg));
    results.push_back(runCuDaoDependencyKernel(cfg, buffers));
    results.push_back(runCuDaoMemset(cfg, buffers));
    results.push_back(runCuDaoMemcpy(cfg, buffers));

    printTable(results);

    destroyBuffers(buffers);
    return 0;
}
