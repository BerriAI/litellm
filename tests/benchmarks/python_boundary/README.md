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
