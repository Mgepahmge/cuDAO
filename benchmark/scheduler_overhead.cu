#include <cuDAO.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdint>
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

    // 用于"kernel 句柄缓存是否命中"对照组的 kernel 变体数量：
    // 固定复用同一个 kernel（现有 cuDAO noop 测试） vs 轮流切换这
    // kKernelVariantCount 个功能等价但地址不同的 kernel。
    constexpr size_t kKernelVariantCount = 4;

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
        if (cfg.copyBytes < sizeof(int)) {
            // memcpy 依赖对照组里的 writeMarkerKernel 需要至少写入一个 int，
            // 冷指针池的地址切分也按 copyBytes 步进，太小会导致越界。
            die("copy-bytes must be at least 4 bytes (needed by the dependency-marker kernel)");
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

    // kernel 句柄缓存对照组用的功能等价 kernel 变体：内容跟 noopKernel
    // 完全一样，唯一目的是拥有不同的函数地址/符号，逼 cuDAO 在每次调用
    // 时看到一个"新的 kernel"。
    __global__ void noopKernelVariant0() { asm volatile(""); }
    __global__ void noopKernelVariant1() { asm volatile(""); }
    __global__ void noopKernelVariant2() { asm volatile(""); }
    __global__ void noopKernelVariant3() { asm volatile(""); }

    __global__ void dependencyKernel(int* out, const int* in) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            out[0] = in[0] + 1;
        }
    }

    // 手工依赖对照组用的最小"生产者" kernel：只写入 buffer 的首个 int，
    // 配合 cudaEventRecord/cudaStreamWaitEvent 在原生 CUDA 里手动建立
    // 跨 stream 依赖，作为 cuDAO 自动依赖追踪的对照基线。
    __global__ void writeMarkerKernel(int* ptr) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            ptr[0] = 1;
        }
    }

    void* toVoidPtr(const CUdeviceptr ptr) {
        return reinterpret_cast<void*>(ptr);
    }

    CUdeviceptr allocOrDie(const size_t bytes, const char* what) {
        CUdeviceptr ptr{};
        const CUresult re = cuMemAlloc(&ptr, bytes);
        if (re != CUDA_SUCCESS) {
            dieDriver(what, re);
        }
        return ptr;
    }

    cudaStream_t streamOrDie(const unsigned int flags, const char* what) {
        cudaStream_t s{};
        const cudaError_t err = cudaStreamCreateWithFlags(&s, flags);
        if (err != cudaSuccess) {
            dieCuda(what, err);
        }
        return s;
    }

    cudaEvent_t eventOrDie(const char* what) {
        cudaEvent_t e{};
        const cudaError_t err = cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            dieCuda(what, err);
        }
        return e;
    }

    struct DeviceBuffers {
        CUdeviceptr noopScratch{};
        CUdeviceptr depIn{};
        CUdeviceptr depOut{};
        CUdeviceptr memsetDst{};
        CUdeviceptr memcpySrc{};
        CUdeviceptr memcpyDst{};
        cudaStream_t baselineStream{};

        // 手工 event 依赖对照组：两个 stream 来回切换 + ping-pong event，
        // 用来复现"在多个内部 stream 之间正确建立一次跨 stream 依赖"的
        // 原生 CUDA 成本，而不是简单地把所有调用堆进同一个 stream，
        // 让硬件 FIFO 顺序免费兜底（那样测不出"建立依赖"这个动作本身
        // 的开销）。
        cudaStream_t depStreamA{};
        cudaStream_t depStreamB{};
        cudaEvent_t depEventA{};
        cudaEvent_t depEventB{};

        cudaStream_t memcpyStreamA{};
        cudaStream_t memcpyStreamB{};
        cudaEvent_t memcpyEventA{};
        cudaEvent_t memcpyEventB{};

        // 冷指针池：整个 benchmark 运行期间（含 warmup）里的每一次调用
        // 都消费池中一个此前从未被 cuDAO 看到过的全新地址，用来验证
        // pointer-tracking 是否存在未被现有 warmup 摊销掉的首次注册开销。
        CUdeviceptr depInColdPool{};
        CUdeviceptr depOutColdPool{};
        CUdeviceptr memcpySrcColdPool{};
        CUdeviceptr memcpyDstColdPool{};
        size_t coldPoolCount{};
    };

    DeviceBuffers createBuffers(const BenchmarkConfig& cfg) {
        DeviceBuffers buffers{};
        const size_t memsetBytes = std::max<size_t>(cfg.copyBytes, sizeof(uint32_t));

        buffers.noopScratch = allocOrDie(sizeof(int), "cuMemAlloc(noopScratch)");
        buffers.depIn = allocOrDie(sizeof(int), "cuMemAlloc(depIn)");
        buffers.depOut = allocOrDie(sizeof(int), "cuMemAlloc(depOut)");
        buffers.memsetDst = allocOrDie(memsetBytes, "cuMemAlloc(memsetDst)");
        buffers.memcpySrc = allocOrDie(cfg.copyBytes, "cuMemAlloc(memcpySrc)");
        buffers.memcpyDst = allocOrDie(cfg.copyBytes, "cuMemAlloc(memcpyDst)");

        buffers.baselineStream = streamOrDie(cudaStreamNonBlocking, "cudaStreamCreateWithFlags(baseline)");

        buffers.depStreamA = streamOrDie(cudaStreamNonBlocking, "cudaStreamCreateWithFlags(depStreamA)");
        buffers.depStreamB = streamOrDie(cudaStreamNonBlocking, "cudaStreamCreateWithFlags(depStreamB)");
        buffers.depEventA = eventOrDie("cudaEventCreateWithFlags(depEventA)");
        buffers.depEventB = eventOrDie("cudaEventCreateWithFlags(depEventB)");

        buffers.memcpyStreamA = streamOrDie(cudaStreamNonBlocking, "cudaStreamCreateWithFlags(memcpyStreamA)");
        buffers.memcpyStreamB = streamOrDie(cudaStreamNonBlocking, "cudaStreamCreateWithFlags(memcpyStreamB)");
        buffers.memcpyEventA = eventOrDie("cudaEventCreateWithFlags(memcpyEventA)");
        buffers.memcpyEventB = eventOrDie("cudaEventCreateWithFlags(memcpyEventB)");

        // 冷指针池大小 = 整个 run 里 submit 会被调用的总次数
        // （warmup 轮次同样要消费全新地址，否则 warmup 阶段本身就
        // 会把这些指针"焐热"）。注意这里内存占用跟 batch/trials/warmup/
        // copy-bytes 都成正比，配置较大时请留意显存是否够用，必要时
        // 通过命令行参数调小。
        buffers.coldPoolCount = (cfg.warmup + cfg.trials) * cfg.batch;
        buffers.depInColdPool = allocOrDie(buffers.coldPoolCount * sizeof(int), "cuMemAlloc(depInColdPool)");
        buffers.depOutColdPool = allocOrDie(buffers.coldPoolCount * sizeof(int), "cuMemAlloc(depOutColdPool)");
        buffers.memcpySrcColdPool = allocOrDie(buffers.coldPoolCount * cfg.copyBytes, "cuMemAlloc(memcpySrcColdPool)");
        buffers.memcpyDstColdPool = allocOrDie(buffers.coldPoolCount * cfg.copyBytes, "cuMemAlloc(memcpyDstColdPool)");

        const int zero = 0;
        auto h2d = [&](const CUdeviceptr dst, const void* src, const size_t bytes, const char* what) {
            const CUresult re = cuMemcpyHtoD(dst, src, bytes);
            if (re != CUDA_SUCCESS) {
                dieDriver(what, re);
            }
        };
        h2d(buffers.depIn, &zero, sizeof(zero), "cuMemcpyHtoD(depIn)");
        h2d(buffers.noopScratch, &zero, sizeof(zero), "cuMemcpyHtoD(noopScratch)");
        h2d(buffers.memcpySrc, &zero, sizeof(zero), "cuMemcpyHtoD(memcpySrc)");

        // 冷指针池整体清零一次即可，逐元素初始化不影响调度开销测试，
        // 且这段初始化发生在计时窗口之外。
        CUresult zeroRe = cuMemsetD32(buffers.depInColdPool, 0, buffers.coldPoolCount);
        if (zeroRe != CUDA_SUCCESS) { dieDriver("cuMemsetD32(depInColdPool)", zeroRe); }
        zeroRe = cuMemsetD32(buffers.depOutColdPool, 0, buffers.coldPoolCount);
        if (zeroRe != CUDA_SUCCESS) { dieDriver("cuMemsetD32(depOutColdPool)", zeroRe); }
        zeroRe = cuMemsetD8(buffers.memcpySrcColdPool, 0, buffers.coldPoolCount * cfg.copyBytes);
        if (zeroRe != CUDA_SUCCESS) { dieDriver("cuMemsetD8(memcpySrcColdPool)", zeroRe); }
        zeroRe = cuMemsetD8(buffers.memcpyDstColdPool, 0, buffers.coldPoolCount * cfg.copyBytes);
        if (zeroRe != CUDA_SUCCESS) { dieDriver("cuMemsetD8(memcpyDstColdPool)", zeroRe); }

        // event 对象在第一次被 cudaStreamWaitEvent 引用前必须至少 record
        // 过一次，否则是未定义行为。这里先各自 record 一次并同步，
        // 让所有 event 在计时开始前处于"已完成"状态。
        cudaError_t evErr = cudaEventRecord(buffers.depEventA, buffers.depStreamA);
        if (evErr != cudaSuccess) { dieCuda("cudaEventRecord(depEventA init)", evErr); }
        evErr = cudaEventRecord(buffers.depEventB, buffers.depStreamB);
        if (evErr != cudaSuccess) { dieCuda("cudaEventRecord(depEventB init)", evErr); }
        evErr = cudaEventRecord(buffers.memcpyEventA, buffers.memcpyStreamA);
        if (evErr != cudaSuccess) { dieCuda("cudaEventRecord(memcpyEventA init)", evErr); }
        evErr = cudaEventRecord(buffers.memcpyEventB, buffers.memcpyStreamB);
        if (evErr != cudaSuccess) { dieCuda("cudaEventRecord(memcpyEventB init)", evErr); }
        evErr = cudaDeviceSynchronize();
        if (evErr != cudaSuccess) { dieCuda("cudaDeviceSynchronize(after event init)", evErr); }

        return buffers;
    }

    void destroyBuffers(DeviceBuffers& buffers) {
        cuMemFree(buffers.memcpyDstColdPool);
        cuMemFree(buffers.memcpySrcColdPool);
        cuMemFree(buffers.depOutColdPool);
        cuMemFree(buffers.depInColdPool);

        cudaEventDestroy(buffers.memcpyEventB);
        cudaEventDestroy(buffers.memcpyEventA);
        cudaStreamDestroy(buffers.memcpyStreamB);
        cudaStreamDestroy(buffers.memcpyStreamA);

        cudaEventDestroy(buffers.depEventB);
        cudaEventDestroy(buffers.depEventA);
        cudaStreamDestroy(buffers.depStreamB);
        cudaStreamDestroy(buffers.depStreamA);

        cudaStreamDestroy(buffers.baselineStream);
        cuMemFree(buffers.memcpyDst);
        cuMemFree(buffers.memcpySrc);
        cuMemFree(buffers.memsetDst);
        cuMemFree(buffers.depOut);
        cuMemFree(buffers.depIn);
        cuMemFree(buffers.noopScratch);
    }

    // ---------------------------------------------------------------
    // 零依赖原生 CUDA 基线
    // ---------------------------------------------------------------

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

    BenchmarkResult runRawCudaMemset(const BenchmarkConfig& cfg, const DeviceBuffers& buffers) {
        return runBenchmark(
            "raw cuda memset",
            cfg,
            [&](size_t) {
                // cudaMemsetAsync 只能按字节填充，跟 cuDAO memset 用的
                // 0x12345678 32 位模式不完全等价；这里比较的是"发起一次
                // 纯 memset 调用"的开销量级，不追求填充值语义一致。
                const cudaError_t err = cudaMemsetAsync(
                    reinterpret_cast<void*>(buffers.memsetDst),
                    0x78,
                    cfg.copyBytes,
                    buffers.baselineStream);
                if (err != cudaSuccess) {
                    dieCuda("cudaMemsetAsync", err);
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

    BenchmarkResult runRawCudaMemcpy(const BenchmarkConfig& cfg, const DeviceBuffers& buffers) {
        return runBenchmark(
            "raw cuda memcpy D2D",
            cfg,
            [&](size_t) {
                const cudaError_t err = cudaMemcpyAsync(
                    reinterpret_cast<void*>(buffers.memcpyDst),
                    reinterpret_cast<const void*>(buffers.memcpySrc),
                    cfg.copyBytes,
                    cudaMemcpyDeviceToDevice,
                    buffers.baselineStream);
                if (err != cudaSuccess) {
                    dieCuda("cudaMemcpyAsync", err);
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

    // ---------------------------------------------------------------
    // 手工建立等价依赖的原生 CUDA 基线
    // 对照 cuDAO 的自动依赖追踪：两个 stream 来回切换 + event 显式同步，
    // 复现"如果开发者自己维护跨 stream 依赖安全性，需要付出多少成本"。
    // ---------------------------------------------------------------

    BenchmarkResult runRawCudaDepKernelEvent(const BenchmarkConfig& cfg, DeviceBuffers& buffers) {
        return runBenchmark(
            "raw cuda dep kernel (event)",
            cfg,
            [&](size_t j) {
                const bool useA = (j % 2 == 0);
                cudaStream_t curStream = useA ? buffers.depStreamA : buffers.depStreamB;
                cudaEvent_t prevEvent = useA ? buffers.depEventB : buffers.depEventA;
                cudaEvent_t curEvent = useA ? buffers.depEventA : buffers.depEventB;

                cudaError_t err = cudaStreamWaitEvent(curStream, prevEvent, 0);
                if (err != cudaSuccess) {
                    dieCuda("cudaStreamWaitEvent", err);
                }

                dependencyKernel<<<kGrid, kBlock, 0, curStream>>>(
                    reinterpret_cast<int*>(toVoidPtr(buffers.depOut)),
                    reinterpret_cast<const int*>(toVoidPtr(buffers.depIn)));
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    dieCuda("dependencyKernel launch", err);
                }

                err = cudaEventRecord(curEvent, curStream);
                if (err != cudaSuccess) {
                    dieCuda("cudaEventRecord", err);
                }
            },
            [&]() {
                cudaError_t err = cudaStreamSynchronize(buffers.depStreamA);
                if (err != cudaSuccess) {
                    dieCuda("cudaStreamSynchronize(depStreamA)", err);
                }
                err = cudaStreamSynchronize(buffers.depStreamB);
                if (err != cudaSuccess) {
                    dieCuda("cudaStreamSynchronize(depStreamB)", err);
                }
            }
        );
    }

    BenchmarkResult runRawCudaMemcpyEvent(const BenchmarkConfig& cfg, DeviceBuffers& buffers) {
        return runBenchmark(
            "raw cuda memcpy D2D (event)",
            cfg,
            [&](size_t j) {
                const bool useA = (j % 2 == 0);
                cudaStream_t curStream = useA ? buffers.memcpyStreamA : buffers.memcpyStreamB;
                cudaEvent_t prevEvent = useA ? buffers.memcpyEventB : buffers.memcpyEventA;
                cudaEvent_t curEvent = useA ? buffers.memcpyEventA : buffers.memcpyEventB;

                cudaError_t err = cudaStreamWaitEvent(curStream, prevEvent, 0);
                if (err != cudaSuccess) {
                    dieCuda("cudaStreamWaitEvent", err);
                }

                // 先手工"生产" src 内容，语义上对齐 cuDAO memcpy 测试里
                // src/dst 两个被追踪资源各自的写依赖。
                writeMarkerKernel<<<kGrid, kBlock, 0, curStream>>>(
                    reinterpret_cast<int*>(toVoidPtr(buffers.memcpySrc)));
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    dieCuda("writeMarkerKernel launch", err);
                }

                err = cudaMemcpyAsync(
                    reinterpret_cast<void*>(buffers.memcpyDst),
                    reinterpret_cast<const void*>(buffers.memcpySrc),
                    cfg.copyBytes,
                    cudaMemcpyDeviceToDevice,
                    curStream);
                if (err != cudaSuccess) {
                    dieCuda("cudaMemcpyAsync", err);
                }

                err = cudaEventRecord(curEvent, curStream);
                if (err != cudaSuccess) {
                    dieCuda("cudaEventRecord", err);
                }
            },
            [&]() {
                cudaError_t err = cudaStreamSynchronize(buffers.memcpyStreamA);
                if (err != cudaSuccess) {
                    dieCuda("cudaStreamSynchronize(memcpyStreamA)", err);
                }
                err = cudaStreamSynchronize(buffers.memcpyStreamB);
                if (err != cudaSuccess) {
                    dieCuda("cudaStreamSynchronize(memcpyStreamB)", err);
                }
            }
        );
    }

    // ---------------------------------------------------------------
    // cuDAO 现有测试项（热路径：固定复用同一批指针/kernel，
    // warmup 阶段已经把它们"焐热"过）
    // ---------------------------------------------------------------

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

    // ---------------------------------------------------------------
    // 缓存命中 / 冷启动对照组
    // ---------------------------------------------------------------

    cuDAO::cuDAOStatus launchRotatingNoopKernel(const size_t idx) {
        switch (idx % kKernelVariantCount) {
            case 0: return cuDAO::launchKernel(noopKernelVariant0, kGrid, kBlock, 0);
            case 1: return cuDAO::launchKernel(noopKernelVariant1, kGrid, kBlock, 0);
            case 2: return cuDAO::launchKernel(noopKernelVariant2, kGrid, kBlock, 0);
            default: return cuDAO::launchKernel(noopKernelVariant3, kGrid, kBlock, 0);
        }
    }

    BenchmarkResult runCuDaoNoopRotatingKernel(const BenchmarkConfig& cfg) {
        // 每次调用轮换使用 kKernelVariantCount 个功能等价但地址不同的
        // kernel。如果 cuDAO 内部按 kernel 函数地址缓存元数据（比如
        // 参数签名解析结果），这里应该比固定同一个 kernel 的
        // "cuDAO noop kernel" 明显更慢；如果两者接近，说明句柄缓存
        // 不是当前开销的主要来源。
        size_t globalCallIndex = 0;
        return runBenchmark(
            "cuDAO noop kernel (rotating kernel)",
            cfg,
            [&](size_t) {
                dieStatus("launchKernel(rotating noop)", launchRotatingNoopKernel(globalCallIndex++));
            },
            [&]() {
                dieStatus("deviceSynchronize", cuDAO::deviceSynchronize());
            }
        );
    }

    BenchmarkResult runCuDaoDependencyKernelCold(const BenchmarkConfig& cfg, DeviceBuffers& buffers) {
        // 整个运行期间（含 warmup）单调递增地消费冷指针池，确保每一次
        // submit 用到的 depIn/depOut 都是 cuDAO 此前从未见过的全新地址，
        // 跟固定复用同一对指针的 "cuDAO dep kernel" 形成对照。
        size_t coldIndex = 0;
        return runBenchmark(
            "cuDAO dep kernel (cold pointer)",
            cfg,
            [&](size_t) {
                const CUdeviceptr in = buffers.depInColdPool + coldIndex * sizeof(int);
                const CUdeviceptr out = buffers.depOutColdPool + coldIndex * sizeof(int);
                ++coldIndex;
                dieStatus("launchKernel(dependencyKernel cold)",
                          cuDAO::launchKernel(
                              dependencyKernel,
                              kGrid,
                              kBlock,
                              0,
                              cuDAO::write(reinterpret_cast<int*>(toVoidPtr(out))),
                              cuDAO::read(reinterpret_cast<int*>(toVoidPtr(in)))
                          ));
            },
            [&]() {
                dieStatus("deviceSynchronize", cuDAO::deviceSynchronize());
            }
        );
    }

    BenchmarkResult runCuDaoMemcpyCold(const BenchmarkConfig& cfg, DeviceBuffers& buffers) {
        size_t coldIndex = 0;
        return runBenchmark(
            "cuDAO memcpy D2D (cold pointer)",
            cfg,
            [&](size_t) {
                const CUdeviceptr src = buffers.memcpySrcColdPool + coldIndex * cfg.copyBytes;
                const CUdeviceptr dst = buffers.memcpyDstColdPool + coldIndex * cfg.copyBytes;
                ++coldIndex;
                dieStatus("cuDAOMemcpy(cold)",
                          cuDAO::cuDAOMemcpy(
                              reinterpret_cast<int*>(toVoidPtr(dst)),
                              reinterpret_cast<const int*>(toVoidPtr(src)),
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

    std::puts("cuDAO scheduler overhead benchmark (extended)");
    std::printf("batch=%zu trials=%zu warmup=%zu copyBytes=%zu\n",
                cfg.batch, cfg.trials, cfg.warmup, cfg.copyBytes);
    std::printf("cold pointer pool: %zu entries per resource (allocated once up front, each entry touched exactly once for the whole run)\n\n",
                buffers.coldPoolCount);

    std::vector<BenchmarkResult> results;

    // 零依赖原生基线
    results.push_back(runRawCudaNoop(cfg, buffers));
    results.push_back(runRawCudaMemset(cfg, buffers));
    results.push_back(runRawCudaMemcpy(cfg, buffers));

    // 手工建立等价依赖的原生基线（对照 cuDAO 的自动依赖追踪）
    results.push_back(runRawCudaDepKernelEvent(cfg, buffers));
    results.push_back(runRawCudaMemcpyEvent(cfg, buffers));

    // cuDAO 现有测试项（热路径）
    results.push_back(runCuDaoNoop(cfg));
    results.push_back(runCuDaoDependencyKernel(cfg, buffers));
    results.push_back(runCuDaoMemset(cfg, buffers));
    results.push_back(runCuDaoMemcpy(cfg, buffers));

    // 缓存命中 / 冷启动对照
    results.push_back(runCuDaoNoopRotatingKernel(cfg));
    results.push_back(runCuDaoDependencyKernelCold(cfg, buffers));
    results.push_back(runCuDaoMemcpyCold(cfg, buffers));

    printTable(results);

    destroyBuffers(buffers);
    return 0;
}
