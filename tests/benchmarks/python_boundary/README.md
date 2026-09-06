# Rust/Python boundary performance

These benchmarks load the installed release extension directly, without importing the Python SDK or calling providers. The default-off Rust `bench` feature adds a private `_bench` namespace; normal wheels do not expose it. The synthetic routes use the same route macro and execution helpers as production

Conversion cases cover both directions, round trips, and the actual typed Messages response serializer. Execution cases cover synchronous calls, awaited asynchronous calls, GIL release/reacquire, four Python caller threads, and async concurrency of 1, 8, and 32. Fixtures cover empty objects, stream chunks, nested Unicode/tool inputs, and image-like strings from 1 KiB to 16 MiB. These protect the boundary, not provider transforms, network latency, or application tail latency

## Local validation

Run from the repository root with the pinned Rust toolchain installed

```bash
uv venv --python 3.12.12 .venv-boundary
uv pip sync --python .venv-boundary/bin/python --require-hashes tests/benchmarks/python_boundary/requirements.txt
.venv-boundary/bin/maturin build --release --locked --features bench,extension-module --out boundary-dist
uv pip install --python .venv-boundary/bin/python --no-deps boundary-dist/*.whl
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-boundary/bin/python -m pytest -c /dev/null --rootdir=. -p no:cacheprovider -p pytest_codspeed.plugin --confcutdir=tests/benchmarks/python_boundary tests/benchmarks/python_boundary/test_boundary.py --codspeed --codspeed-mode=walltime
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-boundary/bin/python -m pytest -c /dev/null --rootdir=. -p no:cacheprovider --confcutdir=tests/benchmarks/python_boundary tests/benchmarks/python_boundary/test_budgets.py
.venv-boundary/bin/python tests/benchmarks/python_boundary/budgets.py measure --output /tmp/boundary-report.json
```

Input construction, runtime initialization, event-loop creation, thread-pool startup, and correctness assertions happen outside timing. Async measurements include dispatch through a reused Python event loop and wait for result delivery. Thread measurements include pool dispatch. Concurrency cases report nanoseconds per completed batch, not individual request latency; their IDs include the concurrency

## Activate the gates

`perf_benchmarks.yml` runs on every PR and integration-branch push. Its explicit jobs cover the existing Python suite, boundary CPU simulation, and boundary walltime. All CodSpeed uploads stay in this workflow so CodSpeed can aggregate them correctly. Shared release-wheel setup lives in the `setup-python-boundary` composite action. Simulation measures conversion CPU cost on Ubuntu. Walltime and the independent fixed-budget collector run on `codspeed-macro`. Runner caches separate architecture, release profile, and benchmark features

Enable public-repository access for the Macro runner group, then dispatch `Calibrate Python boundary budgets` (`perf_python_boundary_calibration.yml`) on the benchmark branch. It can also be called through `CodSpeed Benchmarks` with `calibrate_boundary=true`. That run collects five reports from the same wheel and produces `candidate-budgets.json` in the `python-boundary-calibration` artifact

Review the measurements and copy the candidate to `budgets.json`. Calibration rejects runs whose medians differ by more than 10% from their median. Each ceiling is 120% of the median baseline. The initial empty manifest intentionally fails normal budget checks until calibration is complete; never substitute local laptop measurements

In CodSpeed, set conversion simulation thresholds to 5% and walltime thresholds to 10%, disable informational-only failures, and establish a comparison baseline on `litellm_internal_staging`. In GitHub, require `CodSpeed Performance Analysis`, `Python boundary (simulation)`, and `Python boundary (walltime)` on protected integration branches. Calibration has its own workflow and check names, uploads no CodSpeed results, and cannot satisfy the required budget checks

Before enabling merge protection, verify a trial PR that adds an extra conversion fails CPU analysis, and a delayed async operation fails walltime analysis and the fixed budget check. Remove the trial slowdowns before merging. Repository checks alone do not configure CodSpeed thresholds or branch protection

## Fixed budgets

The independent collector warms each callable, calibrates batches to at least 20 ms, and records 30 samples. Gates compare median nanoseconds per call/batch against the checked-in ceiling; samples and a comparison table are uploaded even when the gate fails

Missing cases, unexpected cases, invalid samples, missing budgets, or changed runtime/build identities fail. Reports and calibrated budgets include the source revision and extension hash; calibration requires identical builds. Python, compiler, dependency locks, profile, and architecture must match the budget environment. Source changes may change the extension hash without invalidating comparison

Budget updates are deliberate reviewed changes, never automatic ratchets. Recalibrate on the same Macro runner when changing the pinned environment. The standard Criterion serialization benchmark remains available through `cargo bench -p litellm-python-bridge --bench serialization` from `litellm-rust`
