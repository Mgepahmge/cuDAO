<p align="center">
  <img src="assets/cuDAO_LOGO.png" alt="cuDAO" width="2000"/>
</p>

> ⚠️ **Early Development** — cuDAO is in early development. APIs are unstable, features are incomplete, and bugs are expected. Use at your own risk.

**Dependency-Aware Ordering** — A header-only CUDA runtime library for automatic concurrent memory access scheduling.

cuDAO transparently manages dependencies between concurrent CUDA kernels based on their memory access semantics. Users annotate kernel arguments with `read()` / `write()` wrappers (or rely on `const T*` / `T*` type inference), and cuDAO automatically inserts the correct synchronization barriers on GPU streams — with zero manual event management.

---

## How It Works

cuDAO uses a dedicated scheduler thread that consumes kernel launch requests from a wait-free MPSC queue. For each kernel, it inspects the declared read/write access patterns and inserts `cuStreamWaitValue64` / `cuStreamWriteValue64` barriers on the selected CUDA stream, ensuring correct ordering without over-serializing independent operations.

---

## Requirements

### Compiler

| Platform | Compiler | Minimum Version                      |
| -------- | -------- | ------------------------------------ |
| Windows  | MSVC     | Visual Studio 2017 15.8 (MSVC 19.15) |
| Linux    | GCC      | GCC 9                                |

Both must support C++17 (`/std:c++17` or `-std=c++17`).

### CUDA Toolkit

| Requirement           | Minimum Version |
| --------------------- | --------------- |
| CUDA Toolkit          | 11.0            |
| nvcc C++17 support    | CUDA 11.0+      |
| `cuStreamWaitValue64` | CUDA 9.0+       |

### GPU Architecture

**Volta (sm_70) or newer is required.**

`cuStreamWaitValue64` and `cuStreamWriteValue64` require `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS`, which is guaranteed on Volta (sm_70) and all subsequent architectures.

| Architecture | Example GPUs                 |
| ------------ | ---------------------------- |
| Volta        | Tesla V100, Titan V          |
| Turing       | RTX 2060/2070/2080, Tesla T4 |
| Ampere       | RTX 3070/3080/3090, A100     |
| Ada          | RTX 4060/4070/4080/4090      |
| Hopper       | H100                         |

### Operating System

| Platform | Requirement                                                   |
| -------- | ------------------------------------------------------------- |
| Linux    | Kernel 2.6.22+ (futex, available on all modern distributions) |
| Windows  | Windows 8 / Windows Server 2012 or newer (`WaitOnAddress`)    |

### Build Tools

| Tool     | Minimum Version | Notes                                       |
| -------- | --------------- | ------------------------------------------- |
| CMake    | 3.25            | Required for configuration and installation |
| Python 3 | 3.x             | Required to generate the single-header file |

---

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build
```

Replace `89` with your GPU's compute capability (e.g. `70` for Volta, `80` for A100, `90` for H100).

### Options

| Option                        | Default | Description                                            |
|-------------------------------|---------|--------------------------------------------------------|
| `CUDAO_BUILD_TESTS`           | ON      | Build test suite                                       |
| `CUDAO_BUILD_EXAMPLES`        | ON      | Build examples                                         |
| `CUDAO_ENABLE_ASAN`           | OFF     | Enable AddressSanitizer (Debug builds only)            |
| `CUDAO_USE_LEAST_TASK_POLICY` | OFF     | Use least-task stream scheduler instead of round-robin |

### Installation

```bash
cmake --install build
```

Installs headers and CMake package config, enabling use via `find_package(cuDAO)` in downstream projects.

### Documentation

The latest generated documentation is available at:

```text
https://mgepahmge.github.io/cuDAO-docs/
```

Versioned documentation is also published for release tags, for example:

```text
https://mgepahmge.github.io/cuDAO-docs/v0.1.1/
```

To build the documentation locally, install Doxygen and Graphviz first:

```bash
sudo apt-get install -y doxygen graphviz
```

Then run:

```bash
./docs/generate_docs.sh
```

The generated HTML entry point is:

```text
build/docs/html/index.html
```

---

## Usage

```cpp
#include <cuDAO.cuh>

__global__ void addKernel(float* c, const float* a, const float* b, int n) { ... }

// allocate tracked device memory
float* a = nullptr;
float* b = nullptr;
float* c = nullptr;

cuDAO::cuDAOMalloc(&a, sizeof(float) * n);
cuDAO::cuDAOMalloc(&b, sizeof(float) * n);
cuDAO::cuDAOMalloc(&c, sizeof(float) * n);

// scheduler-managed memory operations
cuDAO::cuDAOMemset(a, 0.0f, n);
cuDAO::cuDAOMemset(b, 1.0f, n);

// fire-and-forget kernel launch
cuDAO::launchKernel(addKernel, grid, block, 0, write(c), read(a), read(b), n);

// wait until all previous operations on c are complete
cuDAO::sync(c);

// free tracked memory
cuDAO::cuDAOfree(a);
cuDAO::cuDAOfree(b);
cuDAO::cuDAOfree(c);
```

`cuDAOMemcpy` is also scheduler-managed:

```cpp
cuDAO::cuDAOMemcpy(dst, src, bytes, cuDAO::cuDAOMemcpyType::Auto);
```

In `Auto` mode, cuDAO detects host, device, and managed memory pointers through CUDA pointer attributes.

---

## Current Status

| Feature                       | Status                  |
|-------------------------------|-------------------------|
| MPSC lock-free queue          | ✅ Complete              |
| Scheduler thread              | ✅ Complete              |
| Type-inference access mode    | ✅ Complete              |
| `read()` / `write()` wrappers | ✅ Complete              |
| Version-based dependency      | ✅ Complete              |
| `CudaFuture` / `CudaPromise`  | ✅ Complete              |
| `cuDAOfree`                   | ✅ Complete              |
| `sync(ptr)`                   | ✅ Complete              |
| Multi-device support          | 🚧 Not planned for v0.1 |
| CUDA Graph backend            | 🚧 Not planned for v0.1 |

---

## License

MIT License — Copyright (c) 2026 Mgepahmge