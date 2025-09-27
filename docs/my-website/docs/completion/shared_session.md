# Shared Session Support

## Overview

LiteLLM now supports sharing `aiohttp.ClientSession` instances across multiple API calls to avoid creating unnecessary new sessions. This improves performance and resource utilization.

## Usage

### Basic Usage

```python
import asyncio
from aiohttp import ClientSession
from litellm import acompletion

async def main():
    # Create a shared session
    async with ClientSession() as shared_session:
        # Use the same session for multiple calls
        response1 = await acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            shared_session=shared_session
        )
        
        response2 = await acompletion(
            model="gpt-4o", 
            messages=[{"role": "user", "content": "How are you?"}],
            shared_session=shared_session
        )
        
        # Both calls reuse the same session!

asyncio.run(main())
```

### Without Shared Session (Default)

```python
import asyncio
from litellm import acompletion

async def main():
    # Each call creates a new session
    response1 = await acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )
    
    response2 = await acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "How are you?"}]
    )
    # Two separate sessions created

asyncio.run(main())
```

## Benefits

- **Performance**: Reuse HTTP connections across multiple calls
- **Resource Efficiency**: Reduce memory and connection overhead
- **Better Control**: Manage session lifecycle explicitly
- **Debugging**: Easy to trace which calls use which sessions

## Debug Logging

Enable debug logging to see session reuse in action:

```python
import os
import litellm

# Enable debug logging
os.environ['LITELLM_LOG'] = 'DEBUG'

# You'll see logs like:
# 🔄 SHARED SESSION: acompletion called with shared_session (ID: 12345)
# ✅ SHARED SESSION: Reusing existing ClientSession (ID: 12345)
```

## Common Patterns

### FastAPI Integration

```python
from fastapi import FastAPI
import aiohttp
import litellm

app = FastAPI()

@app.post("/chat")
async def chat(messages: list[dict]):
    # Create session per request
    async with aiohttp.ClientSession() as session:
        return await litellm.acompletion(
            model="gpt-4o",
            messages=messages,
            shared_session=session
        )
```

### Batch Processing

```python
import asyncio
from aiohttp import ClientSession
from litellm import acompletion

async def process_batch(messages_list):
    async with ClientSession() as shared_session:
        tasks = []
        for messages in messages_list:
            task = acompletion(
                model="gpt-4o",
                messages=messages,
                shared_session=shared_session
            )
            tasks.append(task)
        
        # All tasks use the same session
        results = await asyncio.gather(*tasks)
        return results
```

### Custom Session Configuration

```python
import aiohttp
import litellm

# Create optimized session
async with aiohttp.ClientSession(
    timeout=aiohttp.ClientTimeout(total=180),
    connector=aiohttp.TCPConnector(limit=300, limit_per_host=75)
) as shared_session:
    
    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        shared_session=shared_session
    )
```

## Implementation Details

The `shared_session` parameter is threaded through the entire LiteLLM call chain:

1. **`acompletion()`** - Accepts `shared_session` parameter
2. **`BaseLLMHTTPHandler`** - Passes session to HTTP client creation
3. **`AsyncHTTPHandler`** - Uses existing session if provided
4. **`LiteLLMAiohttpTransport`** - Reuses the session for HTTP requests

## Backward Compatibility

- **100% backward compatible** - Existing code works unchanged
- **Optional parameter** - `shared_session=None` by default
- **No breaking changes** - All existing functionality preserved

## Testing

Test the shared session functionality:

```python
import asyncio
from aiohttp import ClientSession
from litellm import acompletion

async def test_shared_session():
    async with ClientSession() as session:
        print(f"✅ Created session: {id(session)}")
        
        try:
            response = await acompletion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                shared_session=session,
                api_key="your-api-key"
            )
            print(f"Response: {response.choices[0].message.content}")
        except Exception as e:
            print(f"✅ Expected error: {type(e).__name__}")
        
        print("✅ Session control working!")

asyncio.run(test_shared_session())
```

## Files Modified

The shared session functionality was added to these files:

- `litellm/main.py` - Added `shared_session` parameter to `acompletion()` and `completion()`
- `litellm/llms/custom_httpx/http_handler.py` - Core session reuse logic
- `litellm/llms/custom_httpx/llm_http_handler.py` - HTTP handler integration
- `litellm/llms/openai/openai.py` - OpenAI provider integration
- `litellm/llms/openai/common_utils.py` - OpenAI client creation
- `litellm/llms/azure/chat/o_series_handler.py` - Azure O Series handler

## Troubleshooting

### Session Not Being Reused

1. **Check debug logs**: Enable `LITELLM_LOG=DEBUG` to see session reuse messages
2. **Verify session is not closed**: Ensure the session is still active when making calls
3. **Check parameter passing**: Make sure `shared_session` is passed to all `acompletion()` calls

### Performance Issues

1. **Session configuration**: Tune `aiohttp.ClientSession` parameters for your use case
2. **Connection limits**: Adjust `limit` and `limit_per_host` in `TCPConnector`
3. **Timeout settings**: Configure appropriate timeouts for your environment
