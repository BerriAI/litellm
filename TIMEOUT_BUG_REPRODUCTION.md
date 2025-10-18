# Aiohttp Transport Timeout Bug Reproduction

## Summary

**Bug**: litellm's aiohttp transport does not properly propagate timeout parameters, resulting in `ClientTimeout` being created with all `None` values. This allows requests to hang indefinitely during SSL operations.

**Impact**: Production Dataflow jobs hung for 12+ minutes per request when network/SSL issues occurred, causing complete job failures.

**Root Cause**: `request.extensions.get("timeout", {})` returns an empty dict `{}` instead of timeout configuration, resulting in:
```python
ClientTimeout(
    sock_connect=None,  # {}.get("connect") = None
    sock_read=None,     # {} .get("read") = None
    connect=None,       # {} .get("pool") = None
)
```

## Evidence

### Stack Trace from Production
Dataflow job hung for 717 seconds (12 minutes) stuck at SSL write:

```python
File "litellm/llms/custom_httpx/aiohttp_transport.py", line 207, in handle_async_request
    response = await client_session.request(
...
File "/usr/local/lib/python3.11/ssl.py", line 930, in write
    return self._sslobj.write(data)
```

### Code Location
`litellm/llms/custom_httpx/aiohttp_transport.py:261-273`

The bug occurs when extracting timeout from request extensions:
```python
async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
    timeout = request.extensions.get("timeout", {})  # ← Returns {}!

    # Later creates ClientTimeout with None values:
    timeout=ClientTimeout(
        sock_connect=timeout.get("connect"),  # None
        sock_read=timeout.get("read"),        # None
        connect=timeout.get("pool"),          # None
    )
```

## Reproduction

### Prerequisites
```bash
export VERTEXAI_PROJECT=your-gcp-project
cd litellm
pip install -e .
```

### Step 1: Demonstrate the Bug

Run the reproduction script with diagnostic logging:

```bash
python reproduce_timeout_bug.py
```

**Expected Output:**
```
[TIMEOUT DEBUG] request.extensions: {...}
[TIMEOUT DEBUG] timeout dict: {}
[TIMEOUT DEBUG] ClientTimeout values: {'sock_connect': None, 'sock_read': None, 'connect': None}
```

This proves that despite passing `timeout=30`, aiohttp receives **no timeout configuration**.

### Step 2: Verify the Fix

Run the fix demonstration:

```bash
python demonstrate_fix.py
```

**Expected Output:**
```
Setting: litellm.disable_aiohttp_transport = True
Making a Vertex AI call with timeout=30 seconds...
Note: No [TIMEOUT DEBUG] logs - we're using httpx native transport
```

With aiohttp transport disabled, httpx's native transport correctly propagates timeouts.

## The Fix

### Workaround (Immediate)
```python
import litellm

# Disable aiohttp transport to use httpx native transport
litellm.disable_aiohttp_transport = True
```

### Proper Fix (Required in litellm)

The aiohttp transport needs to handle the case where `request.extensions["timeout"]` is:
1. An `httpx.Timeout` object (needs conversion)
2. Not set at all (should use a default)
3. An integer/float (needs conversion to dict format)

One approach:
```python
async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
    timeout_ext = request.extensions.get("timeout")

    # Convert httpx.Timeout to dict format aiohttp expects
    if isinstance(timeout_ext, httpx.Timeout):
        timeout = {
            "connect": timeout_ext.connect,
            "read": timeout_ext.read,
            "pool": timeout_ext.pool,
        }
    elif isinstance(timeout_ext, (int, float)):
        # Single timeout value for all operations
        timeout = {
            "connect": timeout_ext,
            "read": timeout_ext,
            "pool": timeout_ext,
        }
    else:
        # Default or use what was provided
        timeout = timeout_ext or {}

    # Now create ClientTimeout with proper values
    timeout=ClientTimeout(
        sock_connect=timeout.get("connect", 60),  # Use defaults if missing
        sock_read=timeout.get("read", 60),
        connect=timeout.get("pool", 60),
    )
```

## Related Issues

- #13524 - User reports similar symptoms with Azure OpenAI (620s timeout waits)
- #14895 - Connection timeouts in high-concurrency scenarios
- #12425 - DISABLE_AIOHTTP_TRANSPORT not working for Vertex models (closed)

## Testing

To verify the bug is fixed:

1. Run `reproduce_timeout_bug.py` - should show proper timeout values, not `{}`
2. Test with actual slow/failing endpoint to verify timeout is enforced
3. Verify no indefinite hangs during SSL operations

## Impact

This bug affects:
- All users using Vertex AI/Gemini through litellm
- Any scenario where network/SSL issues cause delays
- Production workloads running on Dataflow, Lambda, etc.

The severity is **critical** because:
- Requests can hang **indefinitely** (observed 12+ minute hangs)
- No error is raised, just silent hanging
- Affects production reliability and cost
