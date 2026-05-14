---
slug: cleaner-release-versions
title: "LiteLLM release versioning is changing: standard names, MINOR for weekly, PATCH for hotfixes"
date: 2026-04-28
authors:
  - yuneng
description: "Dropping `-stable` and `-nightly` suffixes. Weekly releases bump MINOR; PATCH is now reserved for actual hotfixes. Old releases keep their tags forever; new ones start with `1.84.0`."
tags: [release, packaging, docker]
hide_table_of_contents: false
---

LiteLLM release version names are changing. Two pain points have been driving this:

**1. The `-stable` and `-nightly` suffixes aren't standard.**

Versions like `v1.83.3-stable` and `v1.83.0-nightly` don't match PEP 440 (PyPI) or SemVer 2.0 (Docker / Helm) conventions. Users expecting standard version strings get confused, and tooling that classifies versions has to special-case the suffix.

**2. Weekly releases were bumping PATCH, leaving no room for actual hotfixes.**

Under the old model, each scheduled weekly release bumped the PATCH number: `1.83.0` -> `1.83.1` -> `1.83.2` -> `1.83.3`. When a real hotfix was needed for `1.83.3`, the next PATCH (`1.83.4`) was already reserved for the following week's release. The workaround on Docker was `v1.83.3-stable.patch.1` - but PyPI doesn't accept that syntax, so a hotfix that needed both a Docker image and a Python wheel had no clean way to ship.

<!-- truncate -->

## What's new

Starting with **`1.84.0`**:

- **Drop the suffix.** Stable releases are plain PEP 440 / SemVer 2.0: `1.84.0`. Pre-releases use the standard PEP 440 (`1.84.0rc1`, `1.84.0.dev42`) and SemVer (`1.84.0-rc.1`, `1.84.0-dev.42`) shapes for PyPI and Docker respectively.
- **MINOR bumps weekly.** Each scheduled stable bumps the MINOR component: `1.84.0` -> `1.85.0` -> `1.86.0`.
- **PATCH is reserved for hotfixes.** When `1.84.0` needs a fix, it becomes `1.84.1`. Cleanly installs everywhere - `pip install litellm==1.84.1`, `docker pull ghcr.io/berriai/litellm:1.84.1`.

## Side-by-side

| Scenario | Old name | New name |
|---|---|---|
| Weekly scheduled stable | `v1.83.3-stable` | `1.84.0` or `v1.84.0` (Docker) / `1.84.0` (PyPI) |
| Hotfix on the current stable | `v1.83.3-stable.patch.1` (Docker only - no PyPI release) | `1.84.1` or `v1.84.1` (Docker) / `1.84.1` (PyPI) |
| Release candidate | `v1.84.0-rc` | `1.84.0-rc.1` or `v1.84.0-rc.1` (Docker) / `1.84.0rc1` (PyPI) |
| Nightly | `v1.83.0-nightly` | `1.84.0-dev.42` or `v1.84.0-dev.42` (Docker) / `1.84.0.dev42` (PyPI) |

On Docker, every channel is published in both bare (`1.84.0`) and `v`-prefixed (`v1.84.0`) form going forward — both resolve to the same image digest, so existing pins that include the `v` prefix keep working. On PyPI, every channel uses the bare PEP 440 form (`1.84.0`, never `v1.84.0`).

The hotfix row is the meaningful one. Under the old scheme there was no PyPI publication for `v1.83.3-stable.patch.1`. Under the new scheme, hotfixes ship to both registries and PyPI as a normal release.

## Backwards compatibility

Releases that already shipped with the old naming - `v1.83.x-stable`, `v1.83.x-stable.patch.N`, and existing `1.83.x` PyPI versions - **stay on the registries and PyPI forever**. Anything you currently pin to keeps working. The new naming applies to new releases starting `1.84.0`.

If a maintenance patch is needed on a pre-cutover release line (e.g. a fix on `1.83.x` while `1.84.x` is current), that patch may continue to use the old naming for consistency within the line - release notes will call out which format was used. Long-term, all new releases move to the new naming.

## A few things worth knowing

- **The `v` prefix is optional on Docker tags.** Every Docker tag going forward is published in both bare and `v`-prefixed form — `ghcr.io/berriai/litellm:1.84.0` and `ghcr.io/berriai/litellm:v1.84.0` resolve to the same image (same `sha256` digest), and the same applies to release-candidate and dev/nightly tags. Existing pins that include the `v` prefix keep working without change. PyPI versions remain the bare PEP 440 form: `pip install litellm==1.84.0` (not `==v1.84.0`).
- **`litellm-dev`** - there's a separate `litellm-dev` PyPI package and `*-dev` Docker image family for ad-hoc and one-off builds (e.g. testing a fix before it lands in a release). **Not for production use.** Anything pinned to the standard `litellm` package or `ghcr.io/berriai/litellm:*` Docker tags will never accidentally pick up a `litellm-dev` build.
- **`:latest` Docker tag** points to the most recent stable release on each registry, advancing automatically when a new stable ships. For production deployments we still recommend pinning to a content tag (e.g. `:1.84.0`) so deploys are reproducible.
- **Image signing** ([cosign verify](/blog/ci-cd-v2-improvements#verify-docker-image-signatures)) and verification commands continue to work unchanged with the new tag shapes.
