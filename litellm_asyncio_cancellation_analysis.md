# LiteLLM asyncio.CancelledError Handling: Analysis & Solutions

## Problem Summary

LiteLLM catches `asyncio.CancelledError` in Azure handlers and converts them to `AzureOpenAIError`, which breaks user timeout handling with `asyncio.wait_for()`. The user's workaround manually checks if the current task was cancelled, but there are more elegant solutions.

## Root Cause

**Current problematic code** in `litellm/llms/azure/azure.py:443`:
```python
except asyncio.CancelledError as e:
    raise AzureOpenAIError(status_code=500, message=str(e))
```

**Conflicting requirements:**
- LiteLLM wants to catch Azure service cancellations
- Users need external cancellations (timeouts) to propagate properly

## Elegant Solution (Recommended)

Check cancellation source to distinguish external vs internal cancellations:

```python
except asyncio.CancelledError as e:
    current_task = asyncio.current_task()
    if current_task and current_task.cancelled():
        # External cancellation (user timeout) - preserve it
        logging_obj.post_call(
            input=data["messages"],
            api_key=api_key,
            additional_args={"complete_input_dict": data, "cancellation_type": "external"},
            original_response=str(e),
        )
        raise  # Re-raise original CancelledError
    
    # Internal Azure cancellation - convert to AzureOpenAIError
    logging_obj.post_call(
        input=data["messages"],
        api_key=api_key,
        additional_args={"complete_input_dict": data, "cancellation_type": "azure_internal"},
        original_response=str(e),
    )
    raise AzureOpenAIError(status_code=500, message=f"Azure internal cancellation: {str(e)}")
```

## Why This Is More Elegant

1. **Uses asyncio's built-in mechanisms**: `current_task.cancelled()` is the standard way to check
2. **Simple**: One conditional check vs manual task inspection
3. **Robust**: Handles all timeout scenarios (`asyncio.wait_for`, `asyncio.timeout`, manual cancellation)
4. **Preserves intent**: Still catches Azure service cancellations
5. **Standards compliant**: Follows asyncio best practices for library code

## Alternative Solutions

### Enhanced Version (More Robust)
```python
if current_task and (current_task.cancelled() or current_task.cancelling() > 0):
    raise  # Preserve any external cancellation request
```

### Custom Exception Hierarchy
```python
class AzureCancellationError(AzureOpenAIError):
    """Azure service cancellation"""
    pass
```

## Implementation Locations

1. **Primary**: `litellm/llms/azure/azure.py` - `AzureChatCompletion.acompletion()`
2. **Secondary**: `litellm/llms/azure/completion/handler.py` - `AzureTextCompletion.acompletion()`

## Benefits

- ✅ Fixes `asyncio.wait_for()` timeout handling
- ✅ Maintains Azure error handling
- ✅ Better debugging with cancellation type logging  
- ✅ Follows asyncio best practices
- ✅ Minimal code changes required
- ✅ Backward compatible

## Best Practices Applied

Per Python docs: "In almost all situations [CancelledError] must be re-raised" and asyncio library code "might misbehave if a coroutine swallows asyncio.CancelledError."

This solution elegantly balances LiteLLM's Azure error handling needs with proper asyncio cancellation semantics.