# Existing e2e SDK tests

Wires already-existing live-API SDK tests into the matrix instead of writing new parity tests. Selectors point at real test files and folders, such as `tests/ocr_tests/`, rather than individual node IDs, so future tests added to those folders are picked up automatically.
