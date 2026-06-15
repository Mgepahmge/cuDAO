@page contributing Contributing

# Contributing

cuDAO welcomes contributions, but the project uses strict branch protection because it is a public repository backed by a self-hosted GPU runner.

Maintainer contact: `mgepahmge@outlook.com`.

This page explains the contribution model and the expectations for pull requests.

---

## Contribution model

The `main` branch is protected.

Only the maintainer can push directly to `main` or merge pull requests. Contributors should submit changes through pull requests. A pull request is merged only after maintainer review and approval.

This restriction is intentional. It protects:

- the self-hosted GPU runner;
- repository secrets;
- documentation deployment keys;
- release automation;
- the integrity of generated release artifacts.

---

## Before opening a pull request

Please keep pull requests focused.

Good pull requests usually change one of:

- one bug fix;
- one public API improvement;
- one scheduler/internal behavior improvement;
- one documentation improvement;
- one test coverage improvement.

Before opening a pull request, run the relevant local checks when possible:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCUDAO_BUILD_TESTS=ON

cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For documentation changes:

```bash
./docs/generate_docs.sh
```

---

## Header-only development rules

cuDAO is developed as modular headers under:

```text
include/cuDAO/
```

The public include remains:

```cpp
#include <cuDAO.cuh>
```

The generated single header is a distribution artifact, not the primary development source.

When changing headers:

- preserve the ordered include structure;
- avoid introducing circular include dependencies;
- keep public API changes documented with Doxygen comments;
- keep implementation comments close to the scheduler logic they explain;
- run tests so the generated single header remains valid.

---

## Public API documentation

Public API changes should update Doxygen comments near the affected API.

For example, changes to:

- `launchKernel`;
- `launchKernelSync`;
- `read`;
- `write`;
- `sync`;
- `cuDAOMalloc`;
- `cuDAOMallocAsync`;
- `cuDAOMemcpy`;
- `cuDAOMemcpySync`;
- `cuDAOMemset`;
- `cuDAOfree`;

should also update the corresponding API documentation.

---

## Scheduler changes

Scheduler changes should be conservative.

When adding or modifying a task type, describe:

1. which pointers are read;
2. which pointers are written;
3. when write versions are waited on;
4. when write versions are incremented;
5. when read gates are waited on;
6. when read gates are released;
7. when futures are fulfilled;
8. when slots are unregistered or returned to the pool.

Slot lifetime and stream callback ordering are correctness-critical. A slot must not be returned to the pool until all stream work that uses its version memory has completed.

---

## Memory model expectations

cuDAO distinguishes between memory-copy endpoints and scheduler-tracked dependency targets.

Important rules:

- device and managed pointers can be scheduler-tracked;
- ordinary host pointers can be memcpy endpoints;
- ordinary host pointers should not be passed to `sync(ptr)`;
- host-writing copies should be observed with `cuDAOMemcpySync()`;
- managed memory should be detected through managed-pointer attributes, not only memory type.

Changes that affect pointer classification should update both tests and documentation.

---

## Pull request review

Because the repository uses a self-hosted runner, maintainers may choose when and how to run CI for external contributions.

A contributor should not assume that privileged workflows or deployment workflows will run automatically on unreviewed code.

The maintainer may request:

- smaller patches;
- additional tests;
- documentation updates;
- explanation of dependency-ordering changes;
- local test logs.

For sensitive security or runner-related questions, contact the maintainer directly at `mgepahmge@outlook.com`.

---

## Commit style

Use concise conventional-style commit messages when possible.

Examples:

```text
fix: reset slot write version before unregistering pointers
feat: add scheduler-managed memset API
docs: document scheduler dependency model
ci: publish Doxygen documentation on release tags
```

Keep commit messages factual and tied to observable changes.
