# Rust/Python boundary performance

The default-off Rust `bench` feature exposes private probes through `_native._bench`. They exercise production conversion functions, the route macro, and sync/async execution helpers without provider requests. Normal release wheels do not expose these probes

The 45 cases cover conversion in both directions, typed Messages response serialization, round trips, GIL release/reacquire, synchronous calls, awaited asynchronous calls, four Python caller threads, and async concurrency of 1, 8, and 32. Fixtures include empty objects, stream chunks, nested Unicode/tool inputs, and image-like strings from 1 KiB to 16 MiB

`.github/workflows/codspeed.yml` reports conversion benchmarks with CPU simulation and the full suite with walltime. Walltime uses CodSpeed's dedicated Macro runner. CodSpeed compares pull requests against the integration-branch baseline and applies the configured regression threshold

Keep benchmark names stable after the integration-branch baseline is established. CodSpeed identifies them by pytest path and parameter ID

## Local validation

Run from the repository root with the pinned Rust toolchain installed

```bash
uv venv --python 3.12.12 .venv-boundary
uv pip sync --python .venv-boundary/bin/python --require-hashes tests/benchmarks/python_boundary/requirements.txt
PYO3_PYTHON=.venv-boundary/bin/python .venv-boundary/bin/maturin build --release --locked --features bench,extension-module --out boundary-dist
uv pip install --python .venv-boundary/bin/python --no-deps boundary-dist/*.whl
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-boundary/bin/python -m pytest -c /dev/null --rootdir=. -p no:cacheprovider -p pytest_codspeed.plugin --confcutdir=tests/benchmarks/python_boundary tests/benchmarks/python_boundary/test_boundary.py --codspeed --codspeed-mode=walltime
```

The existing Criterion serialization benchmark remains available through `cargo bench -p litellm-python-bridge --bench serialization` from `litellm-rust`

## Merge protection

After the integration branch has a successful baseline, configure the repository's CodSpeed regression threshold and require `CodSpeed Performance Analysis` in branch protection


## Rust feature coverage

CI runs bridge Clippy and tests with default features, `bench`, `trace-parity`, and `bench,trace-parity`. The benchmark echo route disables generated trace wrappers; production routes keep their existing trace behavior. `extension-module` is checked through wheel builds because it changes Python linking

## Reproducible environments and caches

All three benchmark jobs use Python 3.12.12. The Python suite prepares its environment with `uv sync --frozen` and installs pinned benchmark dependencies before collecting or measuring tests. Measurement invokes the prepared interpreter directly, without dependency resolution

Update Python pins in both setup locations together. For the Python suite, copy `mcp` and `a2a-sdk` versions from `uv.lock` when updating their pins. Keep pytest and pytest-codspeed aligned with the boundary requirements

To update boundary dependencies, edit `requirements.in`, regenerate the hash-locked file, then run the local validation above:

```bash
uv pip compile --python-version 3.12.12 --generate-hashes --no-header tests/benchmarks/python_boundary/requirements.in -o tests/benchmarks/python_boundary/requirements.txt
```

The shared Rust setup requires preinstalled Rustup and reads `rust-toolchain.toml`. Cargo caches include workspace crates and dependencies, separated by workload, architecture, and toolchain. The `workspace-v1` cache generation avoids restoring older dependency-only entries. Compare two runs of the same commit to verify restoration and compilation savings; a cold cache must also pass

## Enforcement rollout

After this PR merges normally, verify that `litellm_internal_staging` has a successful CodSpeed baseline with 25 boundary simulation measurements and 45 boundary walltime measurements. Apply a 10% per-benchmark regression threshold to these new measurements, preserving existing Python-suite thresholds. Ensure regression failures are not configured as informational checks

Use a temporary, unmerged PR targeting the integration branch to introduce repeated conversion work in an existing boundary benchmark without changing its identifier. Confirm CodSpeed reports a regression, remove the extra work, and confirm recovery. Close the validation PR after collecting both results

Only after baseline and failure/recovery validation, require `CodSpeed Performance Analysis` on `litellm_internal_staging`, preserving existing protections. Leave `main` unchanged. Verify an unchanged-code rerun passes. The existing `guard-internal-staging` ruleset (ID `17360296`) owns required status checks. Update that rule in place rather than creating a competing protection rule. Repository and CodSpeed administration access are required; an inaccessible classic branch-protection API is not evidence that rulesets are absent

See [threshold configuration](https://codspeed.io/docs/features/customization) and [branch protection](https://codspeed.io/docs/features/performance-checks)
