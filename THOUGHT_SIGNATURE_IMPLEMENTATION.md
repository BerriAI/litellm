# Thought Signature Implementation for OpenAI Client Compatibility

## Overview

This document describes the implementation of thought signature embedding in tool call IDs for Gemini, enabling OpenAI clients to preserve thought signatures across multi-turn conversations without requiring `provider_specific_fields` support.

## Problem Statement

When using OpenAI clients (instead of the LiteLLM SDK) with Gemini's thinking/reasoning capabilities:
- Gemini returns `thoughtSignature` with function calls
- LiteLLM stores these in `provider_specific_fields` of tool calls
- OpenAI clients don't support `provider_specific_fields`
- Thought signatures are lost when the client sends messages back
- This breaks multi-turn reasoning context

## Solution

Embed thought signatures directly in the tool call ID using URL-safe base64 encoding:

```
Format: call_<uuid>__thought__<base64_signature>
```

### Example

**Original ID:** `call_abc123def456`
**Thought Signature:** `CiQB4/H/Xt5tdo+i0VpVKLh7ostYBjJuoS...`
**Encoded ID:** `call_abc123def456__thought__Q2lRQjQvSC9YdDV0ZG8raT...`

## Implementation Details

### 1. Response Transformation (Gemini → OpenAI)

**File:** `litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py`

When Gemini returns a function call with `thoughtSignature`:

```python
@staticmethod
def _encode_tool_call_id_with_signature(
    base_id: str, thought_signature: Optional[str]
) -> str:
    """Encode thought signature into tool call ID"""
    if thought_signature:
        import base64
        encoded_sig = base64.urlsafe_b64encode(thought_signature.encode()).decode()
        return f"{base_id}__thought__{encoded_sig}"
    return base_id
```

The `_transform_parts` method now:
1. Generates a base tool call ID
2. Embeds the thought signature in the ID
3. Also stores in `provider_specific_fields` for backward compatibility

### 2. Request Transformation (OpenAI → Gemini)

**File:** `litellm/litellm_core_utils/prompt_templates/factory.py`

When converting assistant messages back to Gemini format:

```python
def _get_thought_signature_from_tool(tool: dict) -> Optional[str]:
    """Extract thought signature from tool call"""
    # Check provider_specific_fields first (backward compatibility)
    provider_fields = tool.get("provider_specific_fields") or {}
    if isinstance(provider_fields, dict):
        signature = provider_fields.get("thought_signature")
        if signature:
            return signature
    
    # Check tool call ID (OpenAI client compatibility)
    tool_call_id = tool.get("id")
    if tool_call_id and "__thought__" in tool_call_id:
        import base64
        parts = tool_call_id.split("__thought__", 1)
        if len(parts) == 2:
            _, encoded_sig = parts
            try:
                return base64.urlsafe_b64decode(encoded_sig.encode()).decode()
            except Exception:
                pass
    
    return None
```

The `convert_to_gemini_tool_call_invoke` function now:
1. Extracts signatures from both `provider_specific_fields` AND tool call IDs
2. Converts back to Gemini's `thoughtSignature` format

## Usage Example

### OpenAI Client (Now Works!)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://your-litellm-proxy/v1",
    api_key="your-api-key"
)

# First request
response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ],
    tools=[...],
    extra_body={"reasoning_effort": "low"}
)

# Tool call ID now contains embedded thought signature
tool_call_id = response.choices[0].message.tool_calls[0].id
# Example: "call_abc123__thought__Q2lRQjQvSC9YdDV0ZG8raT..."

# Second request - OpenAI client automatically preserves tool_call_id
messages = [
    {"role": "user", "content": "What is the weather in San Francisco?"},
    response.choices[0].message,  # Tool call with embedded signature
    {
        "role": "tool",
        "tool_call_id": tool_call_id,  # Preserved automatically
        "content": "72°F"
    }
]

# LiteLLM automatically extracts signature from ID and sends to Gemini
response2 = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=messages,
    tools=[...],
    extra_body={"reasoning_effort": "low"}
)
```

### LiteLLM SDK (Still Works!)

```python
from litellm import completion

# First request
response = completion(
    model="gemini/gemini-2.0-flash",
    messages=[...],
    tools=[...],
    reasoning_effort="low"
)

# provider_specific_fields still works
signature = response.choices[0].message.tool_calls[0].provider_specific_fields.get("thought_signature")

# Use normally - LiteLLM handles both methods
messages.append(response.choices[0].message)
response2 = completion(
    model="gemini/gemini-2.0-flash",
    messages=messages,
    tools=[...],
    reasoning_effort="low"
)
```

## Backward Compatibility

The implementation maintains full backward compatibility:

1. **LiteLLM SDK clients**: Still use `provider_specific_fields.thought_signature`
2. **OpenAI clients**: Use embedded signatures in tool call IDs
3. **Priority**: `provider_specific_fields` takes priority if both are present
4. **No breaking changes**: Existing code continues to work

## Testing

Comprehensive tests added in:
- `tests/test_litellm/llms/vertex_ai/gemini/test_thought_signature_in_tool_call_id.py`

Test coverage includes:
- Encoding/decoding signatures in tool call IDs
- Response transformation with embedded signatures
- Request transformation extracting from IDs
- Backward compatibility with `provider_specific_fields`
- Priority handling when both methods present
- End-to-end OpenAI client flow
- Parallel tool calls with signatures

## Benefits

1. ✅ **OpenAI Client Compatibility**: No code changes needed in client applications
2. ✅ **Automatic Preservation**: Tool call IDs are automatically preserved by all clients
3. ✅ **Multi-Turn Context**: Gemini maintains reasoning context across conversations
4. ✅ **Backward Compatible**: Existing LiteLLM SDK code continues to work
5. ✅ **No Special Handling**: Developers don't need to manually manage signatures

## Technical Decisions

### Why URL-Safe Base64?

- **No special characters**: Works in any API/JSON context
- **Reversible**: Can decode back to original signature
- **Standard**: Well-supported encoding across platforms

### Why `__thought__` Separator?

- **Unlikely collision**: Unlikely to appear in normal UUIDs
- **Easy to detect**: Simple string matching
- **Human readable**: Clear what it represents

### Why Both Methods?

- **Gradual migration**: LiteLLM SDK users can migrate gradually
- **Flexibility**: Choose the best method for your use case
- **Safety**: Fallback if one method fails

## Files Modified

1. `litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py`
   - Added `THOUGHT_SIGNATURE_SEPARATOR` constant
   - Added `_encode_tool_call_id_with_signature()` to embed signatures in tool call IDs
   - Modified `_transform_parts()` to embed signatures in tool call IDs
   - Note: Thought signatures from Gemini are already base64-encoded, no additional encoding needed

2. `litellm/litellm_core_utils/prompt_templates/factory.py`
   - Modified `_get_thought_signature_from_tool()` to extract signatures from tool call IDs
   - Removed base64 decoding (signatures are already base64-encoded from Gemini)
   - Updated to use `THOUGHT_SIGNATURE_SEPARATOR` constant
   - Updated signature extraction in `convert_to_gemini_tool_call_invoke()`

3. `tests/test_litellm/llms/vertex_ai/gemini/test_thought_signature_in_tool_call_id.py`
   - New comprehensive test suite

## Future Considerations

- Monitor for any edge cases with very long signatures
- Consider adding metrics/logging for signature preservation
- Potentially add configuration option to disable ID embedding if needed

