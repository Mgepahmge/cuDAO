@page design_overview Design Overview

# Design Overview

cuDAO is a header-only CUDA runtime library that schedules CUDA work according to memory access semantics.

The central idea is simple: instead of requiring users to manually wire CUDA events between streams, cuDAO lets users describe which pointer arguments are read and which are written. The scheduler converts that information into stream-side dependency operations.

cuDAO currently schedules:

- CUDA kernel launches;
- pointer-specific synchronization;
- device, host, and unified-memory allocation registration;
- scheduler-managed memory copy;
- scheduler-managed memset;
- scheduler-managed device free and managed-memory unregister.

---

## Architecture

At a high level, cuDAO has four layers:

| Layer | Responsibility |
|------|----------------|
| Public API | Builds task descriptors from user calls such as `launchKernel`, `cuDAOMemcpy`, `cuDAOMemset`, and `sync`. |
| Task queue | Transfers task descriptors from user threads to the scheduler thread. |
| Scheduler | Registers tracked pointers, chooses worker streams, emits dependency waits/writes, and launches CUDA operations. |
| Runtime resources | Owns CUDA streams, version slots, pinned host control memory, device-side version memory, and callback state. |

The public API is intentionally thin. Most calls only build a `TaskDescriptor`, attach optional completion state, and submit the task to the default scheduler.

The scheduler thread performs the dependency-sensitive work. This keeps submission fast and gives cuDAO one place to maintain pointer-to-slot state.

---

## Task submission model

A submitted task carries three kinds of information:

1. the CUDA operation to perform;
2. the pointers read by the task;
3. the pointers written by the task.

For kernel launches, this information comes from `read()` / `write()` wrappers or pointer constness inference:

| Argument form | Scheduler meaning |
|--------------|-------------------|
| `cuDAO::read(ptr)` | task reads `ptr` |
| `cuDAO::write(ptr)` | task writes `ptr` |
| `const T*` | task reads `ptr` |
| `T*` | task writes `ptr` |

For memory operations, the public API assigns the access sets directly. For example:

- host-to-device copy writes the device destination;
- device-to-host copy reads the device source;
- device-to-device copy reads the source and writes the destination;
- memset writes the target pointer;
- free waits for all prior work on the pointer and then unregisters the slot.

---

## Scheduler phases

Most scheduler handlers follow the same two-phase structure.

### Phase 1: reversible host-side setup

The scheduler first performs operations that can still fail without changing CUDA stream state:

- allocate callback data;
- register new pointer slots if needed;
- resolve a CUDA kernel function symbol;
- check whether a version slot is available;
- fill local caches for write slots and read slots.

If a failure occurs here, the task can be abandoned and an asynchronous `cuDAOStatus` can be pushed to the error queue.

### Phase 2: irreversible CUDA stream submission

After all reversible setup succeeds, the scheduler selects a stream and starts emitting CUDA work:

- `cuStreamWaitValue64` waits for required write versions;
- `cuStreamWaitValue64` waits for read gates to open before writes;
- `cuLaunchHostFunc` updates host-side reader bookkeeping;
- CUDA kernel, memcpy, memset, allocation, or free operation is submitted;
- `cuStreamWriteValue64` publishes new write versions;
- a completion callback releases read gates and fulfills optional futures.

Once Phase 2 starts, failures are treated as internal/runtime failures because CUDA stream work may already have been emitted.

---

## Streams and scheduling policy

cuDAO owns a pool of CUDA streams. Each task is assigned one stream.

The default policy is round-robin. A compile-time option enables the least-task policy:

```text
CUDAO_USE_LEAST_TASK_POLICY
```

The least-task policy tracks outstanding tasks per stream and selects the stream with the lowest current load. Completion callbacks notify the policy when a task finishes.

The scheduling policy does not change the correctness model. Correctness comes from pointer-specific version slots and stream memory operations, not from which stream a task uses.

---

## Host callbacks

cuDAO uses CUDA host callbacks to update host-side scheduler metadata at points that are ordered with respect to stream execution.

Callbacks are used for:

- marking reads as active;
- releasing reads when a task completes;
- fulfilling `CudaFuture` promises;
- unregistering pointer slots after free/unregister tasks;
- notifying the least-task policy that a stream task has completed.

This is why future completion reflects scheduler task completion rather than mere host-side submission.

---

## Error reporting

Public APIs return immediate submission errors directly. Errors that happen later on the scheduler thread are reported asynchronously through the scheduler error queue.

Applications can poll:

```cpp
auto err = cuDAO::cuDAOGetLastError();
```

This design avoids throwing from scheduler internals and keeps public APIs `noexcept`.

---

## Important invariants

cuDAO relies on several invariants:

1. A tracked pointer maps to at most one live version slot.
2. A version slot is not returned to the pool until all prior stream work using it has reached the unregister/free callback.
3. A reused slot must start with host-side and device-side write versions reset to zero.
4. A writer must wait for the previous write version and for the read gate to become zero.
5. A reader must wait for the current write version before starting.
6. A host pointer is not a scheduler dependency slot unless it is represented by a CUDA allocation path that cuDAO explicitly tracks.

These invariants keep pointer dependencies local and allow unrelated pointers to progress independently.
