# Before & After Comparison

## The User's Exact Scenario

The user reported this issue when using Claude via LiteLLM with the Responses API:

**Input to their agent**: `company: REI.com`

**Claude's streaming response** included 5 tool calls:
1. `Research_company_web_page`
2. `Infer_company_value_proposition_and_ICP`
3. `Research_company_news`
4. `Search_web_for_information` (for leadership info)
5. `Search_web_for_information` (for customer reviews)

**Next request** (with tool results) **FAILED** with:
```
Expected toolResult blocks at messages.2.content for the following Ids:
tooluse_BcvnPsX2RJ6ZOvx3M--a4A, tooluse_7zFB6eK0TgKQwqJeRdwnZg, tooluse_wkJb8NQkRDaXm-oHeu_Ubg
```

---

## Before Fix: What Was Being Sent to Claude

### Input (Responses API format - from user)
```json
{
  "input": [
    {
      "role": "user",
      "content": [{"text": "company: REI.com", "type": "input_text"}],
      "type": "message"
    },
    {
      "role": "assistant",
      "content": [{"text": "", "type": "output_text"}],
      "type": "message"
    },
    {"type": "function_call", "call_id": "tooluse_gtWtOTKhTnmCQKrs60fSlg", "name": "Research_company_web_page", ...},
    {"type": "function_call", "call_id": "tooluse_1M_saUAKTj-MC8EbawtbPg", "name": "Infer_company_value_proposition_and_ICP", ...},
    {"type": "function_call", "call_id": "tooluse_ynl6q84sSSOVTJ6mfTV5Qw", "name": "Research_company_news", ...},
    {"type": "function_call", "call_id": "tooluse_ts1BIgB5S0aUf8odh72pJw", "name": "Search_web_for_information", ...},
    {"type": "function_call", "call_id": "tooluse_lSiTLF9MTyCKLV_zjK4LpQ", "name": "Search_web_for_information", ...},
    {"type": "function_call_output", "call_id": "tooluse_gtWtOTKhTnmCQKrs60fSlg", "output": "research data"},
    {"type": "function_call_output", "call_id": "tooluse_1M_saUAKTj-MC8EbawtbPg", "output": "targetIcp data"},
    {"type": "function_call_output", "call_id": "tooluse_ynl6q84sSSOVTJ6mfTV5Qw", "output": "sources data"},
    {"type": "function_call_output", "call_id": "tooluse_ts1BIgB5S0aUf8odh72pJw", "output": "leadership data"},
    {"type": "function_call_output", "call_id": "tooluse_lSiTLF9MTyCKLV_zjK4LpQ", "output": "reviews data"}
  ]
}
```

### ❌ BEFORE: What LiteLLM Sent to Claude (WRONG)
```json
{
  "messages": [
    {
      "role": "user",
      "content": "company: REI.com"
    },
    {
      "role": "assistant",
      "content": ""
    },
    {
      "role": "assistant",
      "tool_calls": [{"id": "tooluse_gtWtOTKhTnmCQKrs60fSlg", "type": "function", "function": {...}}]
    },
    {
      "role": "assistant",
      "tool_calls": [{"id": "tooluse_1M_saUAKTj-MC8EbawtbPg", "type": "function", "function": {...}}]
    },
    {
      "role": "assistant",
      "tool_calls": [{"id": "tooluse_ynl6q84sSSOVTJ6mfTV5Qw", "type": "function", "function": {...}}]
    },
    {
      "role": "assistant",
      "tool_calls": [{"id": "tooluse_ts1BIgB5S0aUf8odh72pJw", "type": "function", "function": {...}}]
    },
    {
      "role": "assistant",
      "tool_calls": [{"id": "tooluse_lSiTLF9MTyCKLV_zjK4LpQ", "type": "function", "function": {...}}]
    },
    {"role": "tool", "tool_call_id": "tooluse_gtWtOTKhTnmCQKrs60fSlg", "content": "research data"},
    {"role": "tool", "tool_call_id": "tooluse_1M_saUAKTj-MC8EbawtbPg", "content": "targetIcp data"},
    {"role": "tool", "tool_call_id": "tooluse_ynl6q84sSSOVTJ6mfTV5Qw", "content": "sources data"},
    {"role": "tool", "tool_call_id": "tooluse_ts1BIgB5S0aUf8odh72pJw", "content": "leadership data"},
    {"role": "tool", "tool_call_id": "tooluse_lSiTLF9MTyCKLV_zjK4LpQ", "content": "reviews data"}
  ]
}
```

**Problem**: 6 separate assistant messages! Claude expects at most 1 assistant message per turn.

**Claude's Response**:
```
❌ BedrockException - Expected toolResult blocks at messages.2.content for the following Ids...
```

---

## After Fix: What Is Now Sent to Claude

### ✅ AFTER: What LiteLLM Now Sends to Claude (CORRECT)
```json
{
  "messages": [
    {
      "role": "user",
      "content": "company: REI.com"
    },
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {"id": "tooluse_gtWtOTKhTnmCQKrs60fSlg", "type": "function", "function": {"name": "Research_company_web_page", ...}},
        {"id": "tooluse_1M_saUAKTj-MC8EbawtbPg", "type": "function", "function": {"name": "Infer_company_value_proposition_and_ICP", ...}},
        {"id": "tooluse_ynl6q84sSSOVTJ6mfTV5Qw", "type": "function", "function": {"name": "Research_company_news", ...}},
        {"id": "tooluse_ts1BIgB5S0aUf8odh72pJw", "type": "function", "function": {"name": "Search_web_for_information", ...}},
        {"id": "tooluse_lSiTLF9MTyCKLV_zjK4LpQ", "type": "function", "function": {"name": "Search_web_for_information", ...}}
      ]
    },
    {"role": "tool", "tool_call_id": "tooluse_gtWtOTKhTnmCQKrs60fSlg", "content": "research data"},
    {"role": "tool", "tool_call_id": "tooluse_1M_saUAKTj-MC8EbawtbPg", "content": "targetIcp data"},
    {"role": "tool", "tool_call_id": "tooluse_ynl6q84sSSOVTJ6mfTV5Qw", "content": "sources data"},
    {"role": "tool", "tool_call_id": "tooluse_ts1BIgB5S0aUf8odh72pJw", "content": "leadership data"},
    {"role": "tool", "tool_call_id": "tooluse_lSiTLF9MTyCKLV_zjK4LpQ", "content": "reviews data"}
  ]
}
```

**Fix**: ALL 5 tool calls are now in a SINGLE assistant message!

**Claude's Response**:
```
✅ Successfully processes the request and generates the next response
```

---

## Visual Comparison

### Before (6 assistant messages ❌)
```
User: "company: REI.com"
Assistant: ""
Assistant: [tool_call_1]  ← Extra message (WRONG)
Assistant: [tool_call_2]  ← Extra message (WRONG)
Assistant: [tool_call_3]  ← Extra message (WRONG)
Assistant: [tool_call_4]  ← Extra message (WRONG)
Assistant: [tool_call_5]  ← Extra message (WRONG)
Tool: result_1
Tool: result_2
Tool: result_3
Tool: result_4
Tool: result_5
```

### After (1 assistant message ✅)
```
User: "company: REI.com"
Assistant: "" + [tool_call_1, tool_call_2, tool_call_3, tool_call_4, tool_call_5]  ← Merged!
Tool: result_1
Tool: result_2
Tool: result_3
Tool: result_4
Tool: result_5
```

---

## Code Change Summary

**Before**:
```python
for _input in input:
    chat_completion_messages = transform_input_item(_input)
    messages.extend(chat_completion_messages)  # Each function_call adds separate message
```

**After**:
```python
i = 0
while i < len(input):
    _input = input[i]

    # NEW: Detect consecutive function_calls
    if is_function_call(_input):
        # Collect ALL consecutive function_calls
        function_call_items = []
        while is_function_call(input[i]):
            function_call_items.append(input[i])
            i += 1

        # Create ONE message with ALL tool calls
        tool_calls_list = [create_tool_call(fc) for fc in function_call_items]

        # Merge into existing empty assistant message or create new one
        if last_message_is_empty_assistant:
            last_message["tool_calls"] = tool_calls_list
        else:
            messages.append({"role": "assistant", "tool_calls": tool_calls_list})
    else:
        # Process other message types normally
        messages.extend(transform_input_item(_input))
        i += 1
```

---

## Why This Matters

1. **Enables Complex Agents**: Agents can now make multiple tool calls in one turn without breaking
2. **Claude Compatibility**: Matches Claude's expected message format exactly
3. **No User Code Changes**: Existing code using the Responses API continues to work
4. **Streaming Support**: Works correctly with streaming responses from Claude
