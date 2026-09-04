# Unit tests

Run independently with `uv run python -m tests.rust-python-harness.strategies.unit_tests.runner --plain`. Configure a `unit_suite` for each mapped API in `strategy.json`

The runner combines mapping validation, Python tests in separate verified backend processes, and Cargo tests. It reports missing and ambiguous counterparts. Native Rust tests and existing Python tests stay in their original locations

See [the suite format](../../README.md#configure-cases) for configuration. No complete API mapping is configured yet
