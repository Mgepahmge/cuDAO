@page dependency_model Dependency Model

# Dependency Model

cuDAO orders work using per-pointer version slots.

Each tracked pointer owns a slot containing:

- a device-side write version counter;
- a host-side expected write version;
- a pinned host read gate;
- host-side pending read count.

The device-side write version and pinned host read gate are used with CUDA stream memory operations. This lets different CUDA streams coordinate without requiring a global device synchronization.

---

## Write version

The write version is a monotonically increasing value associated with a tracked pointer slot.

A task that writes a pointer performs this sequence:

1. wait until the device-side write version reaches the slot's current expected value;
2. increment the host-side expected write version;
3. wait until the read gate is zero;
4. launch the write operation;
5. write the incremented version to device memory.

In simplified form:

```text
wait write_version >= expectedWriteVersion
expectedWriteVersion += 1
wait read_gate == 0
launch write operation
write write_version = expectedWriteVersion
```

This makes later tasks able to wait for the write to complete without synchronizing the whole device.

---

## Read dependency

A task that reads a pointer waits for the current write version before its read begins.

For a read-only task:

```text
wait write_version >= expectedWriteVersion
mark read active
launch read operation
mark read complete
```

The read operation does not increment the write version because it does not modify the memory.

---

## Read gate

The read gate prevents writes from starting while earlier reads are active.

The gate uses the following meaning:

| Value | Meaning |
|-------|---------|
| `0` | no active readers; a writer may proceed |
| `1` | one or more readers are active; writers must wait |

When a read task begins, a stream-ordered host callback increments the slot's pending read count and sets the gate to `1`. When the read task completes, another callback decrements the pending read count. If the count reaches zero, the gate is set back to `0`.

Writers wait for:

```text
read_gate == 0
```

before launching their operation.

This allows multiple readers to proceed concurrently while still preventing write-after-read hazards.

---

## Dependency cases

### Write after write

Two writes to the same pointer are ordered by the write version.

```text
write A: publishes version 1
write B: waits for version 1, then publishes version 2
```

### Read after write

A read waits for the current expected write version.

```text
write A: publishes version 1
read B: waits for version 1
```

### Write after read

A writer waits for the read gate to become zero.

```text
read A: sets read_gate = 1
write B: waits for read_gate == 0
read A completion: sets read_gate = 0
write B proceeds
```

### Read after read

Multiple reads of the same pointer can proceed concurrently after the current write version is satisfied.

---

## Kernel task dependency behavior

Kernel tasks may have multiple read and write pointers.

For each write pointer, the scheduler:

- registers or finds the pointer's slot;
- waits for the prior write version;
- increments the expected write version;
- waits for the read gate to be zero;
- publishes the new write version after the kernel launch.

For each read pointer, the scheduler:

- registers or finds the pointer's slot;
- waits for the current expected write version;
- marks the read as active through a host callback;
- releases the read through the completion callback.

The kernel is launched only after all required pointer waits have been emitted on the selected stream.

---

## Memory-copy dependency behavior

Memory copies are represented as scheduler tasks with read/write sets.

| Copy direction | Dependency behavior |
|---------------|---------------------|
| Host to device | writes the device destination |
| Device to host | reads the device source |
| Device to device | reads the source and writes the destination |
| Host to unified | writes the unified destination |
| Unified to host | reads the unified source |
| Unified to unified | reads the source and writes the destination |

Ordinary host pointers are allowed as copy endpoints but are not dependency slots. If the host must observe completion of a host-writing copy, use `cuDAOMemcpySync()` and wait on the returned `CudaFuture`.

---

## Memset dependency behavior

`cuDAOMemset()` is a write task.

The scheduler:

1. registers or finds the destination pointer slot;
2. waits for the previous write version;
3. increments the expected write version;
4. waits for the read gate to be zero;
5. submits the CUDA memset operation;
6. publishes the new write version.

The `count` argument is the number of elements of width `sizeof(U)`, matching CUDA Driver API D8/D16/D32 memset semantics.

---

## Sync dependency behavior

`sync(ptr)` waits for the tracked pointer's current expected write version.

If the pointer has never been tracked, cuDAO allocates and registers a fresh slot and immediately completes the sync. This supports lazy registration, but it also means ordinary host pointers should not be used with `sync(ptr)` as a completion mechanism.

For host-writing copies, use `cuDAOMemcpySync()` instead.

---

## Free and unregister dependency behavior

Free and unregister tasks must not return a slot to the pool until all prior work on that pointer is complete.

A free task waits for:

```text
write_version >= expectedWriteVersion
read_gate == 0
```

Then it submits the CUDA free operation and resets the device-side write version before the callback unregisters the pointer and returns the slot to the pool.

Resetting the device-side write version is required because slots are reused. Without the reset, a new pointer that reuses the same slot could observe a stale device-side version and incorrectly skip dependency waits.
