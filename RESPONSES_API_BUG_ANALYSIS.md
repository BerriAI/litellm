# Responses API Tool Calls Bug Analysis

## Problem Statement

When using the Responses API with Claude via LiteLLM, if a streaming response contains multiple tool calls, the subsequent request (with tool results) fails with:

```
BedrockException - {"message":"Expected toolResult blocks at messages.2.content for the following Ids: tooluse_BcvnPsX2RJ6ZOvx3M--a4A, tooluse_7zFB6eK0TgKQwqJeRdwnZg, tooluse_wkJb8NQkRDaXm-oHeu_Ubg"}
```

## Root Cause

The issue is in `/home/user/litellm/litellm/responses/litellm_completion_transformation/transformation.py`.

### Current Behavior

When transforming Responses API input items to Claude Messages API format, the code processes each `function_call` item individually:

**Input (Responses API format):**
```json
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "", "type": "message"},
  {"type": "function_call", "call_id": "A", "name": "tool1", "arguments": "{...}"},
  {"type": "function_call", "call_id": "B", "name": "tool2", "arguments": "{...}"},
  {"type": "function_call", "call_id": "C", "name": "tool3", "arguments": "{...}"},
  {"type": "function_call_output", "call_id": "A", "output": "result1"},
  {"type": "function_call_output", "call_id": "B", "output": "result2"},
  {"type": "function_call_output", "call_id": "C", "output": "result3"}
]
```

**Current Output (INCORRECT):**
```json
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": ""},
  {"role": "assistant", "tool_calls": [{"id": "A", ...}]},  // Separate message per tool call
  {"role": "assistant", "tool_calls": [{"id": "B", ...}]},  // This is wrong!
  {"role": "assistant", "tool_calls": [{"id": "C", ...}]},  // This is wrong!
  {"role": "tool", "tool_call_id": "A", ...},
  {"role": "tool", "tool_call_id": "B", ...},
  {"role": "tool", "tool_call_id": "C", ...}
]
```

### Expected Behavior

**Expected Output (CORRECT):**
```json
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "", "tool_calls": [
    {"id": "A", ...},
    {"id": "B", ...},
    {"id": "C", ...}
  ]},
  {"role": "tool", "tool_call_id": "A", ...},
  {"role": "tool", "tool_call_id": "B", ...},
  {"role": "tool", "tool_call_id": "C", ...}
]
```

## Code Analysis

### Problem Location

File: `/home/user/litellm/litellm/responses/litellm_completion_transformation/transformation.py`

#### Function: `_transform_response_input_param_to_chat_completion_message` (line 349)

```python
for _input in input:
    chat_completion_messages = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
        input_item=_input
    )
    # ...
    messages.extend(chat_completion_messages)  # Each function_call adds a separate assistant message
```

#### Function: `_transform_responses_api_function_call_to_chat_completion_message` (line 975)

```python
def _transform_responses_api_function_call_to_chat_completion_message(
    function_call: Dict[str, Any],
) -> List[...]:
    """Transform a single function_call into a Chat Completion message"""
    tool_call = ChatCompletionToolCallChunk(...)

    # Creates a NEW assistant message for each function_call
    chat_completion_response_message = ChatCompletionResponseMessage(
        tool_calls=[tool_call],  # Only contains ONE tool call
        role="assistant",
        content=None,
    )

    return [chat_completion_response_message]  # Returns separate message
```

## Solution

We need to modify `_transform_response_input_param_to_chat_completion_message` to:

1. **Detect consecutive `function_call` items** - Group them together
2. **Merge tool calls into a single assistant message** - Combine all tool_calls into one message
3. **Handle existing assistant message** - If there's an empty assistant message right before the function_calls, merge the tool_calls into it

### Implementation Plan

1. **Add a grouping phase** before processing individual items:
   - Scan through input items
   - Identify consecutive `function_call` items
   - Group them together

2. **Modify message construction**:
   - When processing a group of function_calls, create ONE assistant message with multiple tool_calls
   - If the previous message is an assistant message with empty or None content, merge the tool_calls into it

3. **Preserve existing behavior** for single function_calls and other message types

## Files to Modify

1. `/home/user/litellm/litellm/responses/litellm_completion_transformation/transformation.py`
   - Function: `_transform_response_input_param_to_chat_completion_message` (line 349)
   - Add logic to group consecutive function_call items
   - Merge tool calls into a single assistant message

## Test Case

The bug can be reproduced with this input sequence:
- User message
- Empty assistant message
- 3+ function_call items
- Corresponding function_call_output items

The fix should result in:
- User message
- Single assistant message with all tool_calls merged
- Tool messages for each output

## Additional Issue Mentioned by User

The user also mentioned:
> "What I believe to be reasoning messages are coming through as message type, not reasoning"

This is a separate issue related to how reasoning content is being mapped in streaming responses. This should be investigated separately.
