# OpenAI Realtime Handler Tests

## Important Context: `extra_headers` vs `additional_headers`

### Background

There was confusion about the correct parameter name for passing headers to `websockets.connect()`. This README documents the issue for future maintainers.

### Timeline of Changes

1. **Dec 5, 2025** - Changed `extra_headers` → `additional_headers` (commit `8db7f1b8e4`)
2. **Dec 18, 2025** - Changed `extra_headers` → `additional_headers` again (PR #17950, commit `9f88d61d10`)
3. **Jan 15, 2026** - Upgraded `websockets` from 13.1.0 → 15.0.1 (commit `a3cf178e24`, Issue #19089)
4. **Feb 3, 2026** - Changed `additional_headers` → `extra_headers` (commit `67fc9457e5`)
5. **Feb 4, 2026** - Updated tests to match current code (this PR)

### The Issue

**The `websockets` library changed its API between versions:**

- **websockets 13.1.0**: Supported `additional_headers` parameter ✅
- **websockets 15.0.1**: Only supports `extra_headers` parameter ✅

When websockets was upgraded on Jan 15, 2026, the code still used `additional_headers`, which **completely broke the realtime API** for users:

```python
# This raised: TypeError: got an unexpected keyword argument 'additional_headers'
await websockets.connect(url, additional_headers={...})
```

### The Fix

The Feb 3 commit correctly changed the code to use `extra_headers`, which is the **only parameter accepted** by websockets 15.0.1:

```python
# Correct for websockets 15.0.1+
await websockets.connect(url, extra_headers={...})
```

However, the tests weren't updated at that time, causing CI failures.

### Current State (Feb 2026)

✅ **Code uses `extra_headers`** - Correct for websockets 15.0.1+  
✅ **Tests check for `extra_headers`** - Updated to match the working code  
✅ **No risk to users** - This is a bug fix, not a regression

### For Future Maintainers

**If you see tests failing related to `additional_headers` vs `extra_headers`:**

1. Check the websockets version in `requirements.txt`
2. Verify which parameter the websockets library accepts:
   ```python
   import inspect
   import websockets
   print(inspect.signature(websockets.connect))
   ```
3. Ensure both code AND tests use the correct parameter name

**Do NOT change back to `additional_headers`** unless:
- We downgrade to websockets <14.0, OR
- The websockets library re-introduces this parameter (unlikely)

### Reference

- websockets documentation: https://websockets.readthedocs.io/
- websockets changelog: https://websockets.readthedocs.io/en/stable/project/changelog.html
- Issue #19089: Websocket version error (fixed by upgrading to 15.0.1)
