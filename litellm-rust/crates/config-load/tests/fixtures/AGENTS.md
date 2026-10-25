These configs are copied as-is from `tests/unit/proxy/example_config_yaml` and `tests/unit/proxy/test_configs` for now. Keep each of `valid/` and `invalid/` flat so relative `include` paths still resolve

`invalid/config_with_missing_include.yaml` is in Python's `INVALID_FILES`. `invalid/langfuse_config.yaml` passes Python's read-only `get_config` test, but its model has no `litellm_params`, so Python's `load_config` raises `KeyError` on it
