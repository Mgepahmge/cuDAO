@page memory_model Memory Model

# Memory Model

cuDAO separates two concepts:

1. whether a pointer can be used in a CUDA operation;
2. whether a pointer is tracked as a scheduler dependency target.

This distinction is important for host memory and for automatically classified managed memory.

---

## Tracked pointers

A tracked pointer is a pointer that appears in the scheduler slot map and owns a version slot.

Tracked pointers can be synchronized with:

```cpp
cuDAO::sync(ptr);
```

Device and managed memory are the primary tracked pointer kinds. They can appear as kernel arguments, memcpy sources/destinations, memset targets, and free targets.

Pointers allocated through `cuDAOMalloc()` are registered with the scheduler. Pointers allocated outside cuDAO are lazily registered when they first appear in a scheduler-managed operation.

---

## Host pointers

Ordinary host pointers are not scheduler dependency slots.

They may be used as endpoints of memory copy operations:

```cpp
cuDAO::cuDAOMemcpy(device_dst, host_src, bytes);
cuDAO::cuDAOMemcpySync(host_dst, device_src, bytes);
```

However, they should not be used with:

```cpp
cuDAO::sync(host_ptr); // do not use ordinary host pointers this way
```

The scheduler tracks CUDA-side dependency targets. A pageable host pointer has no cuDAO version slot that can safely represent host-side memory visibility.

If a copy writes to host memory and the host needs to observe completion, use the future-returning copy API:

```cpp
auto result = cuDAO::cuDAOMemcpySync(host_dst, device_src, bytes);

if (auto* future = std::get_if<cuDAO::CudaFuture>(&result)) {
    future->wait();
}
```

---

## CUDA pinned host memory

`cuDAOMalloc(..., cuDAOMemKind::Host)` uses CUDA pinned host allocation.

Pinned host allocations can be released through `cuDAOfree()`. They are not the primary target of dependency versioning in the same way as device or managed memory.

For host-visible completion, prefer explicit futures returned by APIs such as `cuDAOMemcpySync()`.

---

## Device memory

Device memory allocated by `cuDAOMalloc(..., cuDAOMemKind::Device)` is allocated with `cuMemAlloc` and registered with the scheduler.

Device memory can be used with:

- `launchKernel`;
- `cuDAOMemcpy`;
- `cuDAOMemcpySync`;
- `cuDAOMemset`;
- `sync`;
- `cuDAOfree`.

When released through `cuDAOfree()`, device memory uses the scheduler-managed asynchronous free path so that the free operation is ordered after all prior work on the pointer.

---

## Managed memory

Managed memory allocated by `cuDAOMalloc(..., cuDAOMemKind::Unified)` is allocated with `cuMemAllocManaged`.

cuDAO treats managed memory as a tracked dependency target. It can be used in kernel, memcpy, memset, sync, and free operations.

Managed memory detection is based on `CU_POINTER_ATTRIBUTE_IS_MANAGED`, not only on `CU_POINTER_ATTRIBUTE_MEMORY_TYPE`. This matters because a managed pointer may report a current memory type such as device memory even though the allocation itself is managed.

When released through `cuDAOfree()`, managed memory is first unregistered from the scheduler and then released with `cuMemFree`.

---

## Asynchronous allocation

`cuDAOMallocAsync()` submits a scheduler-managed asynchronous device allocation task.

The function waits until CUDA has assigned a valid virtual address to the output pointer. This does not mean all later direct external use is automatically safe. If the application needs to use the pointer directly outside cuDAO immediately after allocation, it should wait for scheduler completion:

```cpp
float* ptr = nullptr;

cuDAO::cuDAOMallocAsync(&ptr, bytes);
cuDAO::sync(ptr);
```

Scheduler-submitted operations that depend on the allocated pointer will be ordered through the normal dependency model.

---

## Automatic copy classification

`cuDAOMemcpy(..., cuDAOMemcpyType::Auto)` classifies source and destination pointers using CUDA pointer attributes.

The intended classification priority is:

1. managed memory detection;
2. CUDA memory type detection;
3. pageable host fallback when CUDA reports that the pointer is not a CUDA allocation.

This lets cuDAO support combinations of host, device, and managed memory while keeping host pointers out of scheduler slot tracking.

---

## Practical rules

Use these rules when writing cuDAO code:

| Situation | Recommended API |
|----------|-----------------|
| Wait for device or managed pointer work | `sync(ptr)` |
| Wait for copy into ordinary host memory | `cuDAOMemcpySync(...).wait()` |
| Allocate tracked device memory | `cuDAOMalloc(&ptr, bytes)` |
| Allocate tracked managed memory | `cuDAOMalloc(&ptr, bytes, cuDAOMemKind::Unified)` |
| Submit stream-ordered device allocation | `cuDAOMallocAsync(&ptr, bytes)` |
| Release CUDA device/managed/pinned host memory | `cuDAOfree(ptr)` |

Avoid treating ordinary host pointers as scheduler dependency targets.
