"""
Privacy-scoped usage and stability telemetry for the LiteLLM proxy.

This package must not import anything else from ``litellm`` so it stays
auditable on its own; ``tests/code_coverage_tests/check_telemetry_imports.py``
enforces that.
"""
