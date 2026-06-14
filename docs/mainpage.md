@mainpage cuDAO

# cuDAO

**cuDAO** is a header-only CUDA runtime library for dependency-aware ordering of concurrent GPU work.

Users declare the memory access semantics of kernel arguments with `cuDAO::read()` and `cuDAO::write()`, or rely on pointer constness for simple inference. The scheduler uses those declarations to order kernels and memory operations on CUDA streams without requiring users to manually wire CUDA events.

> cuDAO is in early development. APIs may change before a stable release.

---

## Design model

cuDAO treats each tracked CUDA-accessible pointer as a dependency target. Submitted work is converted into scheduler tasks that record:

- pointers read by the task;
- pointers written by the task;
- optional host-side completion state.

The scheduler assigns tasks to CUDA streams and uses CUDA stream memory operations to wait for prior writes, gate concurrent readers, and publish new write versions. Independent pointers can proceed concurrently, while conflicting tasks are ordered by the scheduler.

---

## Basic kernel launch

```cpp
#include <cuDAO.cuh>

__global__ void add_kernel(float* out, const float* a, const float* b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

void run(float* out, const float* a, const float* b, int n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    cuDAO::launchKernel(
        add_kernel,
        grid,
        block,
        0,
        cuDAO::write(out),
        cuDAO::read(a),
        cuDAO::read(b),
        n
    );

    cuDAO::sync(out);
}
```

---

## Memory operations

cuDAO also provides scheduler-managed memory APIs:

```cpp
float* ptr = nullptr;

cuDAO::cuDAOMalloc(&ptr, sizeof(float) * n);
cuDAO::cuDAOMemset(ptr, 0.0f, n);
cuDAO::sync(ptr);
cuDAO::cuDAOfree(ptr);
```

For copies that write to ordinary host memory, use the future-returning copy API:

```cpp
auto result = cuDAO::cuDAOMemcpySync(host_out, device_in, bytes);

if (auto* future = std::get_if<cuDAO::CudaFuture>(&result)) {
    future->wait();
}
```

Ordinary host pointers are valid as memory-copy endpoints but are not scheduler dependency slots. Do not pass ordinary host pointers to `cuDAO::sync(ptr)`.

---

## Public API

The primary public API group is @ref public_api.

Important entry points include:

- `cuDAO::launchKernel()`
- `cuDAO::launchKernelSync()`
- `cuDAO::read()`
- `cuDAO::write()`
- `cuDAO::sync()`
- `cuDAO::deviceSynchronize()`
- `cuDAO::cuDAOMalloc()`
- `cuDAO::cuDAOMallocAsync()`
- `cuDAO::cuDAOMemcpy()`
- `cuDAO::cuDAOMemcpySync()`
- `cuDAO::cuDAOMemset()`
- `cuDAO::cuDAOfree()`
- `cuDAO::cuDAOGetLastError()`

---

## Local documentation build

```bash
./docs/generate_docs.sh
```

The generated HTML entry point is:

```text
build/docs/html/index.html
```

---

## Design documentation

The Doxygen site also contains design-level documentation for maintainers:

- @ref design_overview
- @ref dependency_model
- @ref memory_model

