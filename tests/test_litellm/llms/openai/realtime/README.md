# OpenAI Realtime Handler Tests

## Important Context: `additional_headers` vs `extra_headers`

### Background

There was confusion about the correct parameter name for passing headers to `websockets.connect()`. This README documents the resolution for future maintainers.

### Timeline of Changes

1. **Dec 5, 2025** - Changed `extra_headers` → `additional_headers` (commit `8db7f1b8e4`)
2. **Dec 18, 2025** - Changed `extra_headers` → `additional_headers` again (PR #17950, commit `9f88d61d10`)
3. **Jan 15, 2026** - Upgraded `websockets` from 13.1.0 → 15.0.1 (commit `a3cf178e24`, Issue #19089)

### The Issue & Resolution

**The `websockets` library changed its API between versions:**

- **websockets < 14.0**: Used `extra_headers` parameter ✅
- **websockets >= 14.0**: Uses `additional_headers` parameter ✅

**LiteLLM uses websockets 15.0.1** (per requirements.txt), which requires `additional_headers`.

### Verification

You can verify the correct parameter name:

```bash
poetry run python -c "import websockets; import inspect; print(inspect.signature(websockets.connect))"
```

This shows: `additional_headers: 'HeadersLike | None' = None` for websockets 15.0.1.

### Current Implementation (Correct)

```python
# ✅ Correct for websockets 15.0.1+
await websockets.connect(url, additional_headers={
    "Authorization": f"Bearer {api_key}",
    "OpenAI-Beta": "realtime=v1"
})
```

### Impact

This is NOT just a test fix - this was a **critical bug** that affected all realtime APIs:
- OpenAI realtime
- Azure realtime
- xAI realtime
- Any pass-through realtime connections

Using `extra_headers` with websockets 15.0.1 resulted in:
```
TypeError: connect() got an unexpected keyword argument 'extra_headers'
```

### For Future Maintainers

If you see test failures related to header parameters:

1. **Check installed websockets version:**
   ```bash
   poetry run python -c "import websockets; print(websockets.__version__)"
   ```

2. **Check requirements.txt** for the specified version

3. **Verify the correct parameter:**
   - websockets >= 14.0: use `additional_headers`
   - websockets < 14.0: use `extra_headers`

4. **Ensure consistency** across all files:
   - `litellm/llms/openai/realtime/handler.py`
   - `litellm/llms/azure/realtime/handler.py`
   - `litellm/llms/custom_httpx/llm_http_handler.py`
   - `litellm/realtime_api/main.py`
   - `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`

**Current Status (Feb 2026):**
- ✅ websockets version: 15.0.1
- ✅ Correct parameter: `additional_headers`
- ✅ All handlers updated and working
