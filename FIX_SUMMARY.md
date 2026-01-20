# Responses API Tool Calls Bug Fix - Summary

## Issue Reproduced ✓

Successfully identified and fixed the issue where multiple function_call items in the Responses API were causing Claude's Messages API to fail with:

```
BedrockException - {"message":"Expected toolResult blocks at messages.2.content for the following Ids: tooluse_BcvnPsX2RJ6ZOvx3M--a4A, tooluse_7zFB6eK0TgKQwqJeRdwnZg, tooluse_wkJb8NQkRDaXm-oHeu_Ubg"}
```

## Root Cause

The problem was in the Responses API to Chat Completion transformation logic:

**File**: `litellm/responses/litellm_completion_transformation/transformation.py`
**Function**: `_transform_response_input_param_to_chat_completion_message` (line 349)

### What Was Happening (BEFORE FIX)

When the Responses API received input with multiple consecutive `function_call` items:

```python
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "", "type": "message"},  # Empty message from streaming
  {"type": "function_call", "call_id": "A", "name": "tool1", ...},
  {"type": "function_call", "call_id": "B", "name": "tool2", ...},
  {"type": "function_call", "call_id": "C", "name": "tool3", ...},
  {"type": "function_call_output", "call_id": "A", "output": "result1"},
  {"type": "function_call_output", "call_id": "B", "output": "result2"},
  {"type": "function_call_output", "call_id": "C", "output": "result3"}
]
```

The transformation created **SEPARATE assistant messages for EACH tool call**:

```python
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": ""},
  {"role": "assistant", "tool_calls": [{"id": "A", ...}]},  # ❌ Separate message
  {"role": "assistant", "tool_calls": [{"id": "B", ...}]},  # ❌ Separate message
  {"role": "assistant", "tool_calls": [{"id": "C", ...}]},  # ❌ Separate message
  {"role": "tool", "tool_call_id": "A", ...},
  {"role": "tool", "tool_call_id": "B", ...},
  {"role": "tool", "tool_call_id": "C", ...}
]
```

But Claude's Messages API **requires all tool calls from the same turn to be in ONE message**:

```python
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "", "tool_calls": [  # ✅ Single message with all tool calls
    {"id": "A", ...},
    {"id": "B", ...},
    {"id": "C", ...}
  ]},
  {"role": "tool", "tool_call_id": "A", ...},
  {"role": "tool", "tool_call_id": "B", ...},
  {"role": "tool", "tool_call_id": "C", ...}
]
```

## The Fix

### Code Changes

Modified `_transform_response_input_param_to_chat_completion_message` to:

1. **Detect consecutive function_call items** - Changed from `for` loop to `while` loop with index tracking
2. **Group them together** - Collect all consecutive function_call items before processing
3. **Create a single assistant message** - Merge all tool calls into one ChatCompletionResponseMessage
4. **Merge into existing empty assistant message** - If there's already an empty assistant message right before the function_calls, merge the tool_calls into it (common in streaming scenarios)

### Key Implementation Details

```python
# NEW: Detect and group consecutive function_call items
if LiteLLMCompletionResponsesConfig._is_input_item_function_call(input_item=_input):
    # Collect ALL consecutive function_call items
    function_call_items: List[Any] = []
    j = i
    while j < len(input) and LiteLLMCompletionResponsesConfig._is_input_item_function_call(input_item=input[j]):
        function_call_items.append(input[j])
        j += 1

    # Create ONE assistant message with ALL tool calls
    tool_calls_list: List[ChatCompletionToolCallChunk] = []
    for idx, func_call in enumerate(function_call_items):
        tool_call = ChatCompletionToolCallChunk(...)
        tool_calls_list.append(tool_call)

    # If previous message is empty assistant message, merge into it
    # Otherwise, create new assistant message
    if last_message_is_empty_assistant:
        last_msg["tool_calls"] = tool_calls_list
    else:
        messages.append(ChatCompletionResponseMessage(
            role="assistant",
            content=None,
            tool_calls=tool_calls_list
        ))
```

## Testing

Created comprehensive unit tests in `tests/test_litellm/responses/test_multiple_function_calls_merging.py`:

### Test Cases

1. **Multiple function_calls merge into single assistant message** ✅
   - Verifies main fix: 3 function_calls → 1 assistant message with 3 tool_calls
   - Checks merging into existing empty assistant message

2. **Function_calls without preceding assistant message** ✅
   - Creates new assistant message when needed

3. **Function_calls with non-empty assistant message** ✅
   - Creates separate assistant message when previous has content

4. **Single function_call behavior unchanged** ✅
   - Regression test: ensures existing behavior still works

## Impact

### What This Fixes

✅ **Streaming Responses with Multiple Tool Calls**: The primary use case - when Claude streams a response that includes multiple tool calls, the subsequent request with tool results will now work correctly.

✅ **Claude Messages API Compatibility**: Ensures LiteLLM's Responses API correctly transforms to Claude's expected format.

✅ **Multi-turn Tool Conversations**: Enables complex agent workflows that require multiple tool calls in a single turn.

### What's Preserved

✅ **Single Tool Call Behavior**: Existing code using single tool calls continues to work unchanged.

✅ **Non-streaming Responses**: All existing non-streaming behavior preserved.

✅ **Other Message Types**: Tool results, regular messages, etc. all work as before.

## Files Modified

1. **litellm/responses/litellm_completion_transformation/transformation.py**
   - Function: `_transform_response_input_param_to_chat_completion_message` (line 349)
   - Added ~85 lines of grouping and merging logic
   - Changed from for-loop to while-loop with index tracking

2. **tests/test_litellm/responses/test_multiple_function_calls_merging.py** (NEW)
   - 4 comprehensive test cases
   - ~290 lines of test code

## Commit

Committed to branch: `claude/fix-litellm-responses-api-pWkQU`

```
commit eb860aa0
fix(responses-api): merge consecutive function_call items into single assistant message

Fixes issue where multiple function_call items in Responses API input were
creating separate assistant messages, causing Claude's Messages API to fail
```

## Additional Issue Mentioned

The user also mentioned:
> "What I believe to be reasoning messages are coming through as message type, not reasoning"

This appears to be a **separate issue** related to how reasoning content is mapped in streaming responses. This should be investigated in a separate fix as it's a different code path.

## Recommendations

1. **Test with Real Claude API**: While the unit tests verify the transformation logic, testing with actual Claude API streaming responses would provide additional validation.

2. **Monitor for Edge Cases**: Watch for any edge cases like:
   - Mixed function_calls and regular messages
   - Function_calls at the start of the conversation
   - Very large numbers of tool calls (though the fix handles this)

3. **Investigate Reasoning Issue**: The second issue about reasoning messages should be tracked and fixed separately.

## Next Steps

To verify the fix works end-to-end:

1. Set up a test with Claude via LiteLLM using the Responses API
2. Make a streaming request that results in multiple tool calls
3. Send the tool results back
4. Verify the request succeeds (should no longer get the "Expected toolResult blocks" error)

The unit tests pass and the code follows the same patterns used elsewhere in the codebase, so this fix should be safe to merge.
