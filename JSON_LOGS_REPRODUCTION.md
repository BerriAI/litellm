# JSON Logs Issue - Reproduction and Findings

## Issue Summary

When `json_logs: true` is configured in litellm_settings, some logs appear as plain text instead of JSON format, particularly:
- Exception tracebacks
- Health check errors
- Database errors
- Various error logs

## Investigation Findings

### What Works ✅
I created several reproducers that show JSON logging **DOES work correctly** in single-process environments:
- `test_json_logs_reproducer.py` - Tests JSON logs with early environment variable
- `test_json_logs_late_enable.py` - Tests enabling JSON logs after import
- `test_logging_handlers.py` - Tests logging handler hierarchy
- `test_proxy_json_logs_reproducer.py` - Simulates exact proxy startup

**All tests show:**
- ✅ Regular error logs are formatted as JSON
- ✅ Exceptions with `.exception()` are formatted as JSON with stacktrace field
- ✅ Module loggers (like `health_check.py`) propagate to root and format as JSON
- ✅ The `_turn_on_json()` function correctly reconfigures loggers

### Suspected Root Cause: Multi-Worker Issue 🎯

The user (dmc) is likely running litellm proxy with multiple workers (gunicorn/uvicorn workers). The probable issue:

**Problem:** `_turn_on_json()` is called in the main process during config loading, but worker processes may not inherit this logging configuration.

**Why this happens:**
1. Main process loads config and calls `_turn_on_json()` in `proxy_cli.py:682`
2. Logging configuration is **per-process**, not shared across workers
3. When gunicorn/uvicorn spawns worker processes, they import litellm modules fresh
4. Workers get default (non-JSON) logging configuration
5. Most logs come from worker processes, not the main process

**Evidence from user logs:**
- "Using json logs. Setting log_config to None." appears in logs
- But subsequent error logs show plain text format like:
  ```
  15:46:06 - LiteLLM:ERROR: vertex_llm_base.py:495 - Failed to load vertex credentials...
  Traceback (most recent call last):
    ...
  ```
- This format matches the default formatter, not JSON formatter

## How to Reproduce

### Prerequisites
```bash
pip install litellm[proxy]
```

### Test 1: Single Process (Works)
```bash
python3 test_proxy_json_logs_reproducer.py
```

**Expected:** All logs after "Using json logs" are JSON formatted ✅

### Test 2: Multi-Worker Scenario (Likely Fails)

Create config file `test_config.yaml`:
```yaml
model_list:
  - model_name: test-model
    litellm_params:
      model: gpt-3.5-turbo
      api_key: fake-key

litellm_settings:
  json_logs: true
```

Run with multiple workers:
```bash
litellm --config test_config.yaml --num_workers 2 --port 4000
```

Then trigger errors (health checks, invalid models, etc.) and check if logs are JSON or plain text.

## Test Scripts Created

1. **test_json_logs_reproducer.py** - Basic JSON logging test with early env var
2. **test_json_logs_late_enable.py** - Tests enabling JSON logs after import
3. **test_logging_handlers.py** - Detailed handler hierarchy analysis
4. **test_proxy_json_logs_reproducer.py** - Comprehensive proxy startup simulation

All scripts demonstrate that JSON logging works correctly in single-process mode.

## Recommended Solution

The `_turn_on_json()` function needs to be called in each worker process, not just the main process. Possible fixes:

1. **Call `_turn_on_json()` in worker initialization hooks** (gunicorn `post_worker_init` hook)
2. **Set `JSON_LOGS` environment variable** before starting the server (so it's set during module import in all processes)
3. **Move json_logs configuration earlier** in the import chain so it's set before any logging happens

## Code Locations

- JSON logging implementation: `litellm/_logging.py`
- Config loading (CLI): `litellm/proxy/proxy_cli.py:672-682`
- Config loading (server): `litellm/proxy/proxy_server.py:2491-2493`
- JsonFormatter class: `litellm/_logging.py:22-41`
- _turn_on_json function: `litellm/_logging.py:169-179`

## Next Steps

1. ✅ Confirmed JSON logging works in single-process mode
2. ⚠️ Need to test with multi-worker setup to confirm the issue
3. 🔧 Implement fix to ensure `_turn_on_json()` is called in all worker processes
4. ✅ Add test case for multi-worker JSON logging

## User Environment Details

From slack conversation:
- LiteLLM version: 1.80.13-1.80.16
- Python: 3.13
- Deployment: Kubernetes
- Config: `json_logs: true` in `litellm_settings`
- Observed: Mix of JSON and plain text logs, especially exceptions
