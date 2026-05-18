# Release Cycle

Litellm Proxy has the following release cycle:

- `1.x.x-dev.N` (nightly): Releases which pass ci/cd (no manual review). Published on PyPI as `1.x.x.devN`.
- `1.x.x-rc.N` (release candidate): Releases which pass ci/cd + [manual review](https://github.com/BerriAI/litellm/discussions/8495#discussioncomment-12180711) + performance testing (pending — being implemented soon) + a 7-day window for early testers to submit issues. Published on PyPI as `1.x.xrcN`.
- `1.x.x` (stable): An `rc` that has passed everything above, then promoted to stable after a second round of manual testing.

In production, we recommend pinning to the latest stable `1.x.x` release.

:::info Versioning changed starting 1.84.0

The `-stable` and `-nightly` suffixes are gone. Stable releases are now plain PEP 440 / SemVer 2.0 (e.g. `1.84.0`), weekly scheduled releases bump the **MINOR** component, and **PATCH** is reserved for hotfixes. Docker publishes both bare (`1.84.0`) and `v`-prefixed (`v1.84.0`) tags pointing to the same image; PyPI uses the bare PEP 440 form (`1.84.0`, never `v1.84.0`). Releases published under the old naming (`v1.83.x-stable`, etc.) stay available forever.

See [LiteLLM release versioning is changing](/blog/cleaner-release-versions) for the full old → new name mapping.

:::

Follow our release notes [here](https://github.com/BerriAI/litellm/releases).


## FAQ

### Is there a release schedule for LiteLLM stable release?

Stable releases come out every week (typically Sunday). Each scheduled stable bumps the MINOR version: `1.84.0` → `1.85.0` → `1.86.0`.

### What is considered a 'minor' bump vs. 'patch' bump?

Starting with `1.84.0` (see [the versioning blog post](/blog/cleaner-release-versions)):

- 'minor' bumps: the regular weekly scheduled stable release (`1.84.0` → `1.85.0`). This is the normal cadence and may include new backward-compatible features or database tables.
- 'patch' bumps: reserved for hotfixes to the current stable (`1.84.0` → `1.84.1`).
- 'major' bumps: break backward compatibility (`1.x.x` → `2.x.x`).

### Enterprise Support

:::info Support model changing — May 18, 2026

As LiteLLM has grown, the current professional support model no longer fits our scale. We're moving to a new model built around clear, predictable communication on when customers can expect support and changes. The model described below is being deprecated, we'll share details on the new system as we finalize it over the next few weeks.

:::

- Stable releases come out every week. Once a new one is available, we no longer provide support for an older one. 
- If there is a MAJOR change (according to semvar conventions - e.g. 1.x.x -> 2.x.x), we can provide support for upto 90 days on the prior stable image. 
