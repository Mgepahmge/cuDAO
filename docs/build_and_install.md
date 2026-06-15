@page build_and_install Build and Installation

# Build and Installation

This page describes how to build cuDAO from source, generate the distributable single header, run tests, build examples, install the package, and build local documentation.

---

## Requirements

cuDAO requires:

- CMake 3.25 or newer;
- a C++17-capable host compiler;
- CUDA Toolkit 11.0 or newer;
- a Volta-class GPU or newer for stream memory operations;
- Python 3 for single-header generation;
- Doxygen and Graphviz for documentation generation.

On Ubuntu, the documentation dependencies can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y doxygen graphviz
```

---

## Configure

A typical release build is:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

Replace `89` with the CUDA architecture matching your GPU.

Common values:

| GPU generation | Architecture |
|---------------|--------------|
| Volta | 70 |
| Ampere A100 | 80 |
| Ampere RTX 30 series | 86 |
| Ada RTX 40 series | 89 |
| Hopper | 90 |

---

## Build

```bash
cmake --build build --parallel
```

During the build, cuDAO generates the distributable single-header file:

```text
build/single_include/cuDAO.cuh
```

The source tree keeps modular development headers under:

```text
include/cuDAO/
```

The generated single header is used as the release/distribution form.

---

## Build options

| Option | Default | Description |
|--------|---------|-------------|
| `CUDAO_BUILD_TESTS` | `ON` when top-level project | Build test suite |
| `CUDAO_BUILD_EXAMPLES` | `ON` when top-level project | Build examples |
| `CUDAO_ENABLE_ASAN` | `OFF` | Enable AddressSanitizer in Debug builds |
| `CUDAO_USE_LEAST_TASK_POLICY` | `OFF` | Use least-task stream scheduling instead of round-robin |

---

## Tests and examples

Tests and examples intentionally use different include targets.

| Target user | CMake target | Header form |
|------------|--------------|-------------|
| Tests | `cuDAO::single` | Generated single header |
| Examples | `cuDAO::cuDAO` | Source-tree modular headers |

Tests compile against the generated header to ensure the release artifact works. Examples compile against the source-tree aggregation header to improve IDE indexing of the modular headers during development.

To run tests:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCUDAO_BUILD_TESTS=ON

cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

---

## Installation

```bash
cmake --install build
```

Installation installs the generated single header and CMake package configuration files.

Downstream projects can use cuDAO with:

```cmake
find_package(cuDAO REQUIRED)
target_link_libraries(your_target PRIVATE cuDAO::cuDAO)
```

Then include:

```cpp
#include <cuDAO.cuh>
```

The installed `cuDAO::cuDAO` target exposes the installed single-header form.

---

## Local documentation build

Build the Doxygen site locally with:

On Linux/macOS:

```bash
./docs/generate_docs.sh
```

On Windows Powershell:

```powershell
powershell -ExecutionPolicy Bypass -File .\docs\generate_docs.ps1
```

The generated entry point is:

```text
build/docs/html/index.html
```

The documentation build scans the modular source headers and Markdown pages under `docs/`.

---

## Single-header generation

The single-header generation step is part of the default build. It expands the source-tree aggregation header:

```text
include/cuDAO.cuh
```

into:

```text
build/single_include/cuDAO.cuh
```

The generated header must not retain project-local includes such as:

```cpp
#include "cuDAO/..."
```

Release and CI workflows validate that the generated header exists and is suitable for distribution.
