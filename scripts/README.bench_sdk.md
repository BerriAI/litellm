# SDK footprint and startup benchmark

Keep the harness in this repository, outside `litellm/`. One invocation measures one explicitly selected source and writes one artifact directory. It never checks out a revision or updates a shared results file, so the same harness can measure older revisions from separate checkouts

## Run locally

Requires Linux or macOS, Python 3.10+, and uv. Select the same exact Python patch version on every comparison runner. The script declares pinned controller dependencies using inline script metadata, without installing LiteLLM's development environment

```bash
uv run --no-project --python 3.11 scripts/bench_sdk.py --local . --output /tmp/sdk-current
```

Exactly one of `--local`, `--package`, and `--wheel` is required. `--local` without a path means the current working directory. `--package` accepts an exact published version, not a range or `latest`. Both package and wheel modes require a compatible binary wheel and never fall back to compiling an sdist

Local mode copies source into a private temporary directory before invoking `pip wheel`, with PEP 517 build isolation enabled. Git checkouts include tracked working changes and untracked files that are not ignored; deleted and ignored files, Git metadata, and old ignored build products are excluded. Internal symlinks stay inside the copy. Non-Git source directories exclude common environment and build directories. Normal build outputs land in the copied source, and Cargo uses a private target directory. This is workspace isolation, not a security sandbox for untrusted build code. Use wheel mode when measuring a release artifact with generated assets absent from the source snapshot

A checkout must have its normal wheel build prerequisites, including Rust when its build backend requires it. Building and resolving dependencies need network access, but neither operation is a latency metric

For a quick smoke check:

```bash
uv run --no-project --python 3.11 scripts/bench_sdk.py \
  --local . --output /tmp/sdk-smoke --samples 3 --install-samples 1
```

For a published package, another checkout, or a prebuilt wheel:

```bash
uv run --no-project --python 3.11 scripts/bench_sdk.py \
  --package 1.98.0 --output /tmp/sdk-published

uv run --no-project --python 3.11 scripts/bench_sdk.py \
  --local /tmp/older-checkout --output /tmp/sdk-older

uv run --no-project --python 3.11 scripts/bench_sdk.py \
  --wheel /tmp/litellm-version-platform.whl --extras proxy --output /tmp/sdk-proxy
```

Use the actual wheel filename in the last command. Extras change the installed dependency set; the workload still exercises SDK completion, not proxy server startup

## Rust build contract

The current root `pyproject.toml` declares `maturin==1.9.4` as the PEP 517 backend and points it at `litellm-rust/crates/python-bridge/Cargo.toml`, with module name `litellm.rust_bridge._native`. Local mode therefore compiles the extension as part of the wheel build before installing or timing anything. Older checkouts use their own declared backend. There is no separate hand-maintained `cargo build` command in this harness

Python build isolation installs the backend, not a pinned Rust compiler or system linker. Those are runner prerequisites. The current Docker builder installs Rust and then installs the package via `uv sync`. CircleCI's Linux setup pins Rust 1.97.1, while its Windows setup uses floating `stable`. The workspace declares `rust-version = "1.88"`, but the bridge does not set `rust-version.workspace = true`. A consistent compiler pin across runners and inheritance of the intended minimum are remaining build reproducibility gaps, outside this benchmark change. [Cargo inheritance rules](https://doc.rust-lang.org/cargo/reference/workspaces.html#the-package-table)

Result metadata identifies the source mode, whether a build occurred, and native files present in the selected wheel. Build output is retained in `run.log`. Cargo's compiler and registry caches may be shared, but compiled target output, source snapshots, Python environments, and measured runtime state are private to each invocation

## What it measures

| Result | Meaning |
| --- | --- |
| `sizes.root_wheel_bytes` | Compressed LiteLLM wheel |
| `sizes.resolved_wheelhouse_bytes` | LiteLLM and every selected dependency wheel, deduplicated by SHA-256 |
| `sizes.installed_delta_bytes` | File bytes after minus before each pristine install, including bytecode and entry points; symlinks excluded |
| `timings.offline_install` | Hash-verified installation and bytecode compilation from the wheelhouse; environment creation excluded |
| `timings.import` | Timer immediately around `import litellm` in each fresh process |
| `timings.configuration` | Public completion arguments and telemetry configuration after import |
| `timings.first_request` | First synchronous, non-streaming completion through a real HTTP client |
| `timings.second_request` | Second completion in the same process, with connection reuse available |
| `timings.import_to_first_response` | Start of import through the first response |
| `timings.launch_to_import` | Parent launch timestamp through the child's import-complete timestamp |
| `timings.launch_to_first_response` | Parent launch timestamp through the child's first response, excluding interpreter shutdown |
| `timings.python_startup_exit` | pyperf command running `python -I -B -c pass` |
| `timings.import_process_exit` | pyperf command running the guarded import probe, including process startup and shutdown |

Durations are seconds. Every timing has raw samples, count, median, mean, sample standard deviation, median absolute deviation, minimum, and maximum. A single sample has no standard deviation. No percentile or statistical significance claim is made from small sample counts

`diagnostics.json` records externally sampled RSS/USS and loaded modules at import, configuration, first response, and second response. These four snapshots come from a separate process with a controller handshake, not a timing run. Module lists are relative to the diagnostic baseline and cumulative. USS is null when unavailable. `importtime.log` is also a separate diagnostic run

## Isolation and repeatability

The controller contains pip, pyperf, psutil, and Pydantic for input validation. Target virtual environments have only the resolved wheel installation, without pip or benchmark packages. Every runtime sample starts a new interpreter with `-I`, an empty working directory, and an allowlisted environment. The current checkout and user site-packages cannot supply the import

pip compiles bytecode during installation. Runtime probes use `-B`, which reads existing bytecode but does not write more. Warmups are discarded and the OS filesystem cache is not flushed. A private home directory starts empty for each invocation and is shared across its samples. This measures process-cold startup with warm filesystem caches, not first-ever token cache initialization or serverless platform startup

The fake provider runs in the controller process on a dynamically assigned loopback port. It returns one fixed response, checks the request, and verifies exactly two requests per workflow probe. The SDK receives dummy credentials, zero retries, a request timeout, and the bundled model cost map setting. A Python audit hook blocks non-loopback socket operations and external DNS lookups; any blocked access fails the benchmark. This is a guard against accidental Python networking, not a security sandbox for native code or untrusted revisions

Import timing preloads only `sys` and `time`, not JSON, HTTP clients, psutil, or pyperf. Process-to-stage times use the system-wide monotonic performance clock shared by processes on the supported platforms. The controller server, audit hook, and minimal probe scaffolding have overhead, so compare like-for-like runs

## Dependencies and backfills

Each run saves the exact wheelhouse, SHA-256 hashes, a hashed `requirements.lock`, dependency-only `constraints.txt`, pip installation reports, and an installed inventory. Dependencies must have binary wheels for the running Python/platform. An unavailable wheel fails explicitly instead of silently compiling an sdist

By default each source resolves its own declared requirements against the current index. That captures dependency changes but does not reconstruct the index as it existed at an old commit. For an implementation comparison, pass the same dependency constraints to both revisions:

```bash
uv run --no-project --python 3.11 scripts/bench_sdk.py \
  --local /tmp/candidate --constraints /tmp/sdk-current/constraints.txt \
  --output /tmp/sdk-candidate-pinned
```

An incompatible constraint fails instead of being relaxed. To replay without dependency downloads, use the saved root wheel as `--wheel` and the saved wheelhouse as `--wheelhouse`. This freezes the artifact universe on that Python/platform. `--package VERSION --wheelhouse PATH` also resolves that exact version from the archive. `--wheelhouse` does not make a local source build offline: the build backend and Rust dependencies can still need downloads. The controller tools must already be available locally

Run multiple invocations against separate checkouts and separate output directories for backfills. Every invocation has private venvs, build output, pip cache, home, and an ephemeral provider port. Existing output paths are refused. Use separate runners for timing comparisons; simultaneous CPU or disk work on one host contaminates the numbers. Revision selection, job matrices, aggregation, and publishing belong in the future GitHub Actions layer

## Output and validation

Standard output contains only the complete JSON result; progress goes to stderr. `result.json` is written atomically only after every measurement succeeds. Failures exit nonzero and retain logs and artifacts, without a success result. Temporary target environments are always removed

The result records Python/platform, target and harness Git revisions and dirty flags, harness file hashes, tool versions, runtime environment settings, wheel metadata, and raw workflow samples. Archive the entire output directory, not just `result.json`

The pyperf files can be inspected with `pyperf check`, `pyperf stats`, and `pyperf compare_to`. Treat local smoke results as verification that the harness works, not evidence of a regression. Use repeated runs on the same idle runner before setting any thresholds

Run the focused tests in a controller environment with the same pinned packages:

```bash
uv run --no-project --with pip==26.2.1 --with pyperf==2.10.0 --with psutil==7.2.2 --with pydantic==2.13.4 \
  python -m unittest discover -s scripts -p test_bench_sdk.py -v
```

The tests use a small synthetic package to verify the harness without downloading LiteLLM dependencies. Real measurements always use the supplied LiteLLM wheel

This initial suite does not measure streaming/async paths, proxy boot, every provider, native allocation profiles, online installation latency, or cloud platform startup

Method references: [pip repeatable installs](https://pip.pypa.io/en/stable/topics/repeatable-installs/), [pip managing a separate interpreter](https://pip.pypa.io/en/stable/topics/python-option/), [pyperf command](https://pyperf.readthedocs.io/en/latest/cli.html#pyperf-command)
