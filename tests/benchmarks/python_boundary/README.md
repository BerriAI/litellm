# Rust/Python boundary performance

The default-off Rust `bench` feature exposes private probes through `_native._bench`. They exercise production conversion functions, the route macro, and sync/async execution helpers without provider requests. Normal release wheels do not expose these probes

The 45 cases cover conversion in both directions, typed Messages response serialization, round trips, GIL release/reacquire, synchronous calls, awaited asynchronous calls, four Python caller threads, and async concurrency of 1, 8, and 32. Fixtures include empty objects, stream chunks, nested Unicode/tool inputs, and image-like strings from 1 KiB to 16 MiB

Benchmark names describe the operation and payload, for example `test_python_to_rust[image-1KiB]`, `test_rust_to_python[stream-chunk]`, and `test_async_bridge[messages-tools]`. Image sizes describe the synthetic image string, not the entire object. Names containing `batch` measure all calls completing, including dispatch. Keep these names stable once baselines are established because CodSpeed identifies benchmarks by their pytest paths and parameter IDs

## One reporting workflow

`.github/workflows/codspeed.yml` has three explicit jobs: the existing Python suite, boundary CPU simulation, and boundary walltime. All results go to CodSpeed for history, profiling, and PR comparisons. CodSpeed requires one reporting workflow to aggregate its results correctly

The boundary jobs install a pinned release wheel using `setup-python-boundary`. CPU simulation runs on Ubuntu; walltime runs on `codspeed-macro`. Runtime, fixture, event-loop, and worker initialization happen outside measurement. Async cases wait for result delivery through a reused event loop. Concurrent cases include dispatch overhead and report the cost of a completed batch, not individual request latency

## Local validation

Run from the repository root with the pinned Rust toolchain installed

```bash
uv venv --python 3.12.12 .venv-boundary
uv pip sync --python .venv-boundary/bin/python --require-hashes tests/benchmarks/python_boundary/requirements.txt
.venv-boundary/bin/maturin build --release --locked --features bench,extension-module --out boundary-dist
uv pip install --python .venv-boundary/bin/python --no-deps boundary-dist/*.whl
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-boundary/bin/python -m pytest -c /dev/null --rootdir=. -p no:cacheprovider -p pytest_codspeed.plugin --confcutdir=tests/benchmarks/python_boundary tests/benchmarks/python_boundary/test_boundary.py --codspeed --codspeed-mode=walltime --boundary-report=/tmp/boundary-report.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-boundary/bin/python -m pytest -c /dev/null --rootdir=. -p no:cacheprovider --confcutdir=tests/benchmarks/python_boundary tests/benchmarks/python_boundary/test_budgets.py
```

The existing Criterion serialization benchmark remains available through `cargo bench -p litellm-python-bridge --bench serialization` from `litellm-rust`

## Fixed ceilings on the same measurements

The session hook exports the median walltime statistics produced by the pinned `pytest-codspeed` plugin. CodSpeed receives the same benchmark measurements. There is no separate timer, benchmark execution, or calibration workflow

`budgets.py check` compares those exported measurements against `budgets.json`. It fails for exceeded ceilings, missing/unexpected cases, invalid statistics, or mismatched runtime/build environments. The workflow uploads the report and comparison table even when checks fail

Establish initial ceilings from five normal CodSpeed walltime reports for the same build on the Macro runner. Download their `python-boundary-walltime` artifacts into separate directories, then run:

```bash
.venv-boundary/bin/python tests/benchmarks/python_boundary/budgets.py baseline /tmp/run-{1,2,3,4,5}/report.json --output /tmp/candidate-budgets.json
```

Review the candidate and copy it to `budgets.json`. This utility rejects runs whose medians differ by more than 10% from their median and proposes ceilings at 120% of the baseline. It only reads existing reports; it does not measure or upload anything. Initial empty budgets deliberately fail until reviewed ceilings exist. Never substitute laptop measurements

Reports and budgets retain source/build provenance. Environment changes require explicit baseline review. Budget increases are reviewed changes, never automatic ratchets

## Activate merge protection

Set CodSpeed's boundary thresholds to 5% for simulation and 10% for walltime, and disable informational-only failures. After the workflow lands and the integration branch has a comparison baseline, require `Python boundary (simulation)`, `Python boundary (walltime)`, and `CodSpeed Performance Analysis` in branch protection

Verify a trial slowdown fails the performance checks and cannot merge before treating enforcement as active. The code does not configure repository permissions or CodSpeed settings. These benchmarks protect boundary overhead, not provider/network latency or all production workloads
