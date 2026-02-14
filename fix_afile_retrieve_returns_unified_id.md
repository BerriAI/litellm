# Fix: afile_retrieve returns raw provider ID for batch output files

## Bug

`managed_files.afile_retrieve()` Case 2 (file_object already in DB) returned the stored `file_object` without replacing `.id` with the unified file ID. Case 3 (fetch from provider) did this correctly at line 1028.

## Fix

One-line change in `enterprise/litellm_enterprise/proxy/hooks/managed_files.py`:

```python
# Before (line 1013-1014)
if stored_file_object and stored_file_object.file_object:
    return stored_file_object.file_object

# After
if stored_file_object and stored_file_object.file_object:
    stored_file_object.file_object.id = file_id
    return stored_file_object.file_object
```

## Test

```bash
poetry run pytest tests/test_litellm/enterprise/proxy/test_afile_retrieve_returns_unified_id.py -s -vvvv
```

## Test failure (before fix)

```
FAILED tests/test_litellm/enterprise/proxy/test_afile_retrieve_returns_unified_id.py::test_should_return_unified_id_when_file_object_exists_in_db
AssertionError: afile_retrieve should return the unified ID 'bGl0ZWxsbV9wcm94eTp1bmlmaWVkX291dHB1dF9maWxl', but got raw provider ID 'batch_20260214-output-file-1'
assert 'batch_20260214-output-file-1' == 'bGl0ZWxsbV9wcm94eTp1bmlmaWVkX291dHB1dF9maWxl'
=================== 1 failed, 1 retried in 102.95s ===================
```

## Test pass (after fix)

```
tests/test_litellm/enterprise/proxy/test_afile_retrieve_returns_unified_id.py::test_should_return_unified_id_when_file_object_exists_in_db PASSED
============================== 1 passed in 0.11s ===============================
```

## Files changed

- `enterprise/litellm_enterprise/proxy/hooks/managed_files.py` — one-line fix
- `tests/test_litellm/enterprise/proxy/test_afile_retrieve_returns_unified_id.py` — new test
