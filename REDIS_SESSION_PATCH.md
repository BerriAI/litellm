# Redis Session Patch for LiteLLM Issue #12364

## Overview
This is a **temporary patch** to fix the Responses API conversation context timing issue while waiting for the official fix from LiteLLM maintainers.

**Problem**: Conversation context fails on consecutive requests due to 10-second batch processing delay.  
**Solution**: Immediate Redis storage with database fallback.

## Strategy

### Core Principle: Minimal, Non-Breaking Changes
- **Redis-first**: Store session data immediately in Redis after response generation
- **Database fallback**: Keep existing batch processing as backup (resilient to Redis failures)
- **Patch approach**: Minimal code changes, clearly marked as temporary fix
- **Zero breaking changes**: Existing functionality preserved

### Architecture
```
Request → Response Generated → [PATCH] Store in Redis immediately → Return Response
                                        ↓
Later Request → [PATCH] Check Redis first → Found? Use immediately
                                          ↓
                              Not found? → Use existing database/enterprise logic
```

## Implementation

### Files to Modify

#### 1. `/litellm/responses/litellm_completion_transformation/transformation.py`

**Add Redis helper functions** (at the end of file):

```python
# =============================================================================
# PATCH: Redis Session Storage for Issue #12364
# This is a temporary fix for conversation context timing issues
# TODO: Remove when upstream fixes batch processing timing
# =============================================================================

async def _patch_store_session_in_redis(response_id: str, session_id: str, messages: List[Dict]):
    """PATCH: Store session immediately in Redis to avoid batch processing delay"""
    try:
        from litellm.proxy.proxy_server import redis_client
        import json
        
        if redis_client is None:
            return  # No Redis - graceful fallback to existing logic
            
        session_data = {
            "messages": messages,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store with 24-hour TTL
        await redis_client.setex(
            f"litellm_patch:session:{response_id}",
            86400,  # 24 hours
            json.dumps(session_data)
        )
        
    except Exception:
        # PATCH: Silent fail - don't break existing functionality
        pass

async def _patch_get_session_from_redis(previous_response_id: str) -> Optional[Dict]:
    """PATCH: Get session from Redis if available"""
    try:
        from litellm.proxy.proxy_server import redis_client
        import json
        
        if redis_client is None:
            return None
            
        # Decode response ID to get actual request ID
        actual_request_id = ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(
            previous_response_id
        )
        
        # Get session data from Redis
        session_json = await redis_client.get(f"litellm_patch:session:{actual_request_id}")
        
        if session_json:
            return json.loads(session_json)
            
        return None
        
    except Exception:
        # PATCH: Silent fail - fallback to existing logic
        return None
```

**Modify `async_responses_api_session_handler`** (replace existing function):

```python
@staticmethod
async def async_responses_api_session_handler(
    previous_response_id: str,
    litellm_completion_request: dict,
) -> dict:
    """
    Async hook to get the chain of previous input and output pairs and return a list of Chat Completion messages
    
    PATCH: Added Redis-first lookup to fix conversation context timing issues
    """
    
    # PATCH: Try Redis first for immediate availability
    redis_session = await _patch_get_session_from_redis(previous_response_id)
    if redis_session:
        _messages = litellm_completion_request.get("messages") or []
        session_messages = redis_session.get("messages") or []
        litellm_completion_request["messages"] = session_messages + _messages
        litellm_completion_request["litellm_trace_id"] = redis_session.get("session_id")
        return litellm_completion_request
    
    # PATCH: Fallback to existing enterprise/database logic
    if _ENTERPRISE_ResponsesSessionHandler is not None:
        chat_completion_session = ChatCompletionSession(
            messages=[], litellm_session_id=None
        )
        if previous_response_id:
            chat_completion_session = await _ENTERPRISE_ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
                previous_response_id=previous_response_id
            )
        _messages = litellm_completion_request.get("messages") or []
        session_messages = chat_completion_session.get("messages") or []
        litellm_completion_request["messages"] = session_messages + _messages
        litellm_completion_request[
            "litellm_trace_id"
        ] = chat_completion_session.get("litellm_session_id")
    
    return litellm_completion_request
```

#### 2. `/litellm/responses/litellm_completion_transformation/handler.py`

**Modify `async_response_api_handler`** (add one line after response generation):

```python
async def async_response_api_handler(
    self,
    litellm_completion_request: dict,
    request_input: Union[str, ResponseInputParam],
    responses_api_request: ResponsesAPIOptionalRequestParams,
    **kwargs,
) -> Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]:

    previous_response_id: Optional[str] = responses_api_request.get(
        "previous_response_id"
    )
    if previous_response_id:
        litellm_completion_request = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
            previous_response_id=previous_response_id,
            litellm_completion_request=litellm_completion_request,
        )
    
    acompletion_args = {}
    acompletion_args.update(kwargs)
    acompletion_args.update(litellm_completion_request)

    litellm_completion_response: Union[
        ModelResponse, litellm.CustomStreamWrapper
    ] = await litellm.acompletion(
        **acompletion_args,
    )

    if isinstance(litellm_completion_response, ModelResponse):
        responses_api_response: ResponsesAPIResponse = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                chat_completion_response=litellm_completion_response,
                request_input=request_input,
                responses_api_request=responses_api_request,
            )
        )

        # PATCH: Store session immediately in Redis to avoid batch processing delay
        if responses_api_response.id:
            session_id = kwargs.get("litellm_trace_id") or str(uuid.uuid4())
            current_messages = litellm_completion_request.get("messages", [])
            await LiteLLMCompletionResponsesConfig._patch_store_session_in_redis(
                response_id=responses_api_response.id,
                session_id=session_id,
                messages=current_messages
            )

        return responses_api_response

    elif isinstance(litellm_completion_response, litellm.CustomStreamWrapper):
        return LiteLLMCompletionStreamingIterator(
            litellm_custom_stream_wrapper=litellm_completion_response,
            request_input=request_input,
            responses_api_request=responses_api_request,
        )
```

## Required Imports

Add to top of `/litellm/responses/litellm_completion_transformation/transformation.py`:

```python
# PATCH: Additional imports for Redis session storage
from datetime import datetime
import uuid
from typing import Optional, Dict, List
```

## Configuration

**Optional**: Add environment variable control (add to proxy server config):

```python
# PATCH: Redis session storage configuration
REDIS_SESSION_PATCH_ENABLED = os.getenv("REDIS_SESSION_PATCH_ENABLED", "true").lower() == "true"
REDIS_SESSION_PATCH_TTL = int(os.getenv("REDIS_SESSION_PATCH_TTL", "86400"))  # 24 hours
```

## Testing

### Verification Steps
1. **Before patch**: Two consecutive requests fail without 10-second delay
2. **After patch**: Two consecutive requests work immediately
3. **Redis failure**: Still works (falls back to existing logic)
4. **Different models**: Works with Gemini, Claude, etc.

### Test Commands
```bash
# Test 1: Immediate consecutive requests (should work)
curl -X POST http://localhost:4000/v1/responses -d '{"model": "gemini-pro", "input": "Who is Michael Jordan?"}'
# Get response_id from above, then immediately:
curl -X POST http://localhost:4000/v1/responses -d '{"model": "gemini-pro", "input": "Tell me more about him", "previous_response_id": "RESPONSE_ID"}'

# Test 2: Redis failure resilience
# Stop Redis, test should still work (with database fallback)
```

## Rollback Plan

To remove the patch:
1. Remove the `_patch_*` functions from `transformation.py`
2. Revert `async_responses_api_session_handler` to original version
3. Remove the Redis storage line from `handler.py`
4. Clear Redis keys: `redis-cli DEL litellm_patch:session:*`

## Notes

- **Minimal impact**: Only 3 small changes to existing files
- **Graceful degradation**: Works without Redis, falls back to existing logic
- **Temporary**: Designed to be easily removed when upstream fixes the issue
- **Performance**: Redis lookup is faster than database batch processing
- **Memory**: 24-hour TTL prevents Redis memory bloat

---

**This is a temporary patch. Monitor LiteLLM releases for official fix and remove this patch when resolved.**