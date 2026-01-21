# Fix for json_logs Issue

## Problem
When `json_logs: true` is enabled in litellm proxy configuration, some logs were printed as plain text with ANSI color codes instead of being formatted as JSON.

## Root Cause
Several `print()` statements in `proxy_cli.py` and `proxy_server.py` were outputting logs directly to stdout instead of using the configured logger. These bypassed the JsonFormatter entirely.

## Solution
Replaced `print()` statements with `verbose_proxy_logger.info()` calls in the following locations:

### Files Changed:
1. **litellm/proxy/proxy_cli.py**
   - Line 137: `print(f"Using log_config: {log_config}")` → `verbose_proxy_logger.info(f"Using log_config: {log_config}")`
   - Line 140: `print("Using json logs. Setting log_config to None.")` → `verbose_proxy_logger.info("Using json logs. Setting log_config to None.")`

2. **litellm/proxy/proxy_server.py**
   - Lines 2425-2427: Replaced print statement for success callbacks
   - Lines 2443-2445: Replaced print statement for failure callbacks
   - Lines 2675-2679: Replaced print statements for model list initialization
   - Removed ANSI color codes (`\033[94m`, `\033[32m`, etc.) from all messages

## Changes in Detail

### Before:
```python
print(f"{blue_color_code} Initialized Success Callbacks - {litellm.success_callback} {reset_color_code}")
```

### After:
```python
verbose_proxy_logger.info(f"Initialized Success Callbacks - {litellm.success_callback}")
```

### Model List Logging:
**Before:** Printed header + one line per model
```python
print("\033[32mLiteLLM: Proxy initialized with Config, Set models:\033[0m")
for model in model_list:
    print(f"\033[32m    {model.get('model_name', '')}\033[0m")
```

**After:** Single log line with all models
```python
model_names = [model.get('model_name', '') for model in model_list]
verbose_proxy_logger.info(f"LiteLLM: Proxy initialized with Config, Set models: {', '.join(model_names)}")
```

## Impact
- ✅ All startup INFO logs now use the configured JsonFormatter when `json_logs: true`
- ✅ No more ANSI color codes in logs
- ✅ Logs are now parseable by JSON log aggregators (Elasticsearch, Splunk, etc.)
- ✅ ERROR logs with exceptions already worked correctly (unchanged)

## Testing
Run the reproducer script to verify:
```bash
./reproduce_json_logs.sh
```

Expected: All logs after "Using json logs" should be in JSON format with no ANSI color codes.
