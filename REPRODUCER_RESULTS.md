# JSON Logs Reproducer - Results and Analysis

## Summary

**Status**: ✅ **Issue Reproduced Successfully with Real Proxy Server**

When `json_logs: true` is enabled in litellm proxy configuration, there is a **MIX of JSON and non-JSON formatted logs**.

## Reproduction Steps

1. **Start litellm proxy with json_logs enabled:**
   ```bash
   # Using the provided config file
   python3 start_proxy.py
   ```

2. **Make requests to trigger errors:**
   ```bash
   curl -X POST http://localhost:4000/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-1234567890" \
     -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "test"}]}'
   ```

3. **Check logs** in `actual_proxy_logs.txt`

## Findings

### ✅ What WORKS (JSON Formatted)

**ERROR logs with exceptions ARE properly formatted as JSON:**

```json
{"message": "litellm.proxy.proxy_server._handle_llm_api_exception(): Exception occured - litellm.AuthenticationError: ...", "level": "ERROR", "timestamp": "2026-01-19T22:51:46.250125", "stacktrace": "Traceback (most recent call last):\n  File \"/home/user/litellm/litellm/llms/openai/openai.py\", line 840, in acompletion\n..."}
```

```json
{"message": "Failed to load vertex credentials. Check to see if credentials containing partial/invalid information. Error: No module named 'google'", "level": "ERROR", "timestamp": "2026-01-19T22:51:49.318104", "stacktrace": "Traceback (most recent call last):\n..."}
```

### ❌ What DOESN'T WORK (Plain Text)

**Some startup INFO logs are NOT formatted as JSON:**

1. **Plain text message:**
   ```
   Using json logs. Setting log_config to None.
   ```

2. **ANSI colored plain text logs:**
   ```
   [94m Initialized Success Callbacks - [] [0m
   [94m Initialized Failure Callbacks - [] [0m
   [32mLiteLLM: Proxy initialized with Config, Set models:[0m
   [32m    gpt-3.5-turbo[0m
   [32m    azure-gpt-35[0m
   [32m    vertex-gemini[0m
   [32m    azure-embedding[0m
   ```

3. **ASCII art banner** (plain text, not JSON)

## Root Cause

The issue occurs because:

1. ✅ `_turn_on_json()` IS being called correctly (we see "Using json logs" message)
2. ✅ The JsonFormatter IS working for ERROR logs with exceptions
3. ❌ Some code paths use `print()` statements or colored formatters instead of the JSON logger
4. ❌ ANSI color codes (`[94m`, `[32m`, etc.) indicate use of the colored formatter, not JsonFormatter

### Specific Problem Locations

Looking at the colored logs, these are likely coming from:
- `litellm/proxy/proxy_server.py` - proxy initialization messages
- `litellm/proxy/proxy_cli.py` - "Using json logs" print statement (line 140)
- Various INFO level logs that use colored formatters

The colored formatter code in `litellm/_logging.py:95-100`:
```python
formatter = logging.Formatter(
    "\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s",
    datefmt="%H:%M:%S",
)
```

These ANSI escape codes (`\033[92m`, `\033[0m`) are what we see as `[92m` and `[0m` in the logs.

## Test Files Created

1. **reproduce_json_logs_config.yaml** - Proxy config with json_logs enabled and models that trigger errors
2. **start_proxy.py** - Script to start the proxy programmatically
3. **reproduce_json_logs.sh** - Automated test script that starts proxy, makes requests, analyzes logs
4. **actual_proxy_logs.txt** - Captured logs showing the mix of JSON and non-JSON

## Comparison: Expected vs Actual

### Expected (ALL JSON):
```json
{"message": "Using json logs. Setting log_config to None.", "level": "INFO", "timestamp": "..."}
{"message": " Initialized Success Callbacks - []", "level": "INFO", "timestamp": "..."}
{"message": "LiteLLM: Proxy initialized with Config, Set models: gpt-3.5-turbo, azure-gpt-35, vertex-gemini, azure-embedding", "level": "INFO", "timestamp": "..."}
{"message": "litellm.proxy.proxy_server._handle_llm_api_exception():...", "level": "ERROR", "timestamp": "...", "stacktrace": "..."}
```

### Actual (MIXED):
```
Using json logs. Setting log_config to None.
[94m Initialized Success Callbacks - [] [0m
[32mLiteLLM: Proxy initialized with Config, Set models:[0m
{"message": "litellm.proxy.proxy_server._handle_llm_api_exception():...", "level": "ERROR", "timestamp": "...", "stacktrace": "..."}
```

## Severity Assessment

**Impact**: Medium to High for production deployments

- ✅ Critical ERROR logs with exceptions ARE properly formatted
- ❌ INFO/DEBUG logs from startup and initialization are NOT formatted
- ❌ Logs cannot be reliably parsed by JSON log aggregators (Elasticsearch, Splunk, etc.)
- ❌ ANSI color codes appear in logs, breaking JSON parsers

## Recommendations

1. **Remove `print()` statements** in proxy_cli.py and replace with logger calls
2. **Ensure all logging goes through the configured loggers**, not direct print statements
3. **Check for hardcoded colored formatters** that bypass json_logs setting
4. **Add integration test** to verify ALL logs are JSON when json_logs=true

## Files for Review

All reproducer files have been created and are ready for commit:
- `reproduce_json_logs_config.yaml`
- `start_proxy.py`
- `reproduce_json_logs.sh`
- `actual_proxy_logs.txt` (example output)

This reproduces the exact issue that dmc reported in the Slack thread.
