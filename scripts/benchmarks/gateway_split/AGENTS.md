# Benchmark tooling

Run the contract tests with `python -m unittest discover -s scripts/benchmarks/gateway_split -p 'test_*.py'`

Use `run.py` for a short live infrastructure smoke check. Full performance sweeps run through `cloud.py` on separate otherwise idle machines. Never benchmark several topologies concurrently on the same host

All performance measurements use the pinned Bifrost suite and saved patch. Generated trials, logs and local tools stay under ignored results/ and runtime/ directories. Do not commit laptop performance claims as cloud results
