@page release_process Release Process

# Release Process

This page describes the release and documentation publication process for cuDAO.

cuDAO uses a protected `main` branch, GPU validation on a self-hosted runner, tag-gated release publication, and a separate documentation repository for generated Doxygen output.

Maintainer contact: `mgepahmge@outlook.com`.

---

## Branch protection policy

The repository is public and uses a self-hosted GPU runner. For that reason, the `main` branch is strictly protected.

Project policy:

- only the maintainer can push directly to `main`;
- contributors cannot directly push to `main`;
- pull requests cannot be merged by contributors;
- pull requests are merged only after maintainer review and approval;
- unreviewed external changes must not gain access to privileged CI paths, deployment keys, or repository secrets.

This policy protects the self-hosted runner and release/documentation deployment credentials.

---

## Main-branch GPU CI

The GPU CI workflow runs on updates to `main`.

Its purpose is to validate that the current main-branch state still builds and passes the CUDA test suite on the self-hosted runner.

The workflow performs:

1. checkout;
2. toolchain reporting;
3. CMake configuration;
4. build;
5. generated single-header existence check;
6. test execution with `ctest`.

This workflow is the release gate. Release and documentation workflows only publish artifacts for commits that have already passed the main-branch GPU CI.

---

## Release tags

Releases are created from `v*` tags.

Recommended flow:

```bash
git push origin main

# Wait until cuDAO GPU CI succeeds for this commit.

git tag v0.1.1
git push origin v0.1.1
```

Do not tag a commit that has not been pushed to `main` and validated by GPU CI.

If a tag was pushed prematurely and the release/documentation workflow failed, remove and recreate it after fixing the issue:

```bash
git tag -d v0.1.1
git push origin :refs/tags/v0.1.1

git tag v0.1.1
git push origin v0.1.1
```

---

## Single-header release workflow

The single-header release workflow runs only for `v*` tags or manual dispatch.

Before publishing, it verifies:

1. the tag exists;
2. the tagged commit is reachable from `origin/main`;
3. a successful `cuDAO GPU CI` run exists for the same commit on `main`.

After those checks, it:

1. sets up Python;
2. generates `dist/cuDAO.cuh`;
3. validates that the generated header no longer contains project-local includes;
4. packages release assets;
5. creates or updates the GitHub Release.

Release assets include:

| Asset | Description |
|-------|-------------|
| `cuDAO.cuh` | Generated single public header |
| `cuDAO-vX.Y.Z-single-header.tar.gz` | Packaged single-header distribution with README and LICENSE |
| `SHA256SUMS` | Checksums for release assets |

Release notes are generated automatically by GitHub through `gh release create --generate-notes`.

---

## Documentation publication workflow

The documentation workflow uses the same release gate as the single-header release workflow.

It runs only for `v*` tags or manual dispatch and verifies:

1. the tag exists;
2. the tagged commit is reachable from `origin/main`;
3. a successful `cuDAO GPU CI` run exists for the same commit on `main`.

After those checks, it:

1. installs Doxygen, Graphviz, and rsync;
2. runs `docs/generate_docs.sh`;
3. clones the private documentation repository through an SSH deploy key;
4. publishes the generated HTML into versioned and latest documentation directories;
5. pushes the documentation repository.

The published layout is:

```text
/
  index.html
  .nojekyll
  latest/
  v0.1.1/
  v0.1.2/
```

The `latest/` directory is overwritten on each documentation release. Version directories are intended to remain stable.

---

## Required repository settings

The source repository should define these Actions secrets and variables:

| Name | Type | Purpose |
|------|------|---------|
| `DOCS_DEPLOY_KEY` | Secret | Private SSH key used to push generated docs |
| `DOCS_REPO` | Variable | Documentation repository, for example `Mgepahmge/cuDAO-docs` |
| `DOCS_BRANCH` | Variable | Documentation branch, usually `main` |

The documentation repository should add the matching public key as a deploy key with write access.

---

## Security notes

Do not expose release or documentation deployment secrets to unreviewed code.

Because this project uses a public repository and a self-hosted runner, the branch protection and maintainer-controlled merge policy are part of the security model. Any workflow that has access to secrets, deploy keys, or the self-hosted GPU runner should be treated as privileged.
