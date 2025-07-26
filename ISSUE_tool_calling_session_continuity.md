# Issue: Tool/Function Calling Fails with Session Continuity on Gemini Models

## Problem Description

When using `previous_response_id` for session continuity with Gemini models, tool/function calling fails with an error about mismatched function call/response parts. This issue does NOT occur with OpenAI models.

## Error Message

```
BadRequestError: 400 data: {"error": {"message": "litellm.BadRequestError: VertexAIException BadRequestError - b'{\\n  \"error\": {\\n    \"code\": 400,\\n    \"message\": \"Please ensure that the number of function response parts is equal to the number of function call parts of the function call turn.\",\\n    \"status\": \"INVALID_ARGUMENT\"\\n  }\\n}\\n'", "type": null, "param": null, "code": "400"}}
```

## Reproduction Steps

### Request 1: Initial request with tool
```json
{
  "temperature": 0,
  "model": "carto::gemini-2.0-flash",
  "input": [{"role": "user", "content": "What date is it?"}],
  "tools": [{
    "name": "get_date",
    "description": "Get current date in 'YYYY-MM-DD' format.\n",
    "parameters": {"type": "object", "properties": {}},
    "type": "function"
  }],
  "metadata": {"charge_multiplier": "1", "ai_feature": "date"}
}
```

### Response 1: Tool call response
```json
{
  "type": "response.completed",
  "response": {
    "id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOmdlbWluaTttb2RlbF9pZDo4NzU5YzhkMC02NGIxLTRkYTUtYWUxZS1mOWY1Nzg0MTU3NmM7cmVzcG9uc2VfaWQ6ZHFsNGFMdk1HTTI1cXNNUDVaQ1I0QWs=",
    "output": [
      {
        "type": "message",
        "id": "dql4aLvMGM25qsMP5ZCR4Ak",
        "status": "stop",
        "role": "assistant",
        "content": [{"type": "output_text", "annotations": []}]
      },
      {
        "arguments": "{}",
        "call_id": "call_489952e5-9b3b-41af-90fb-1c3482858d06",
        "name": "get_date",
        "type": "function_call",
        "id": "call_489952e5-9b3b-41af-90fb-1c3482858d06",
        "status": "completed"
      }
    ]
  }
}
```

### Request 2: Function result with previous_response_id
```json
{
  "temperature": 0,
  "model": "carto::gemini-2.0-flash",
  "input": [{
    "type": "function_call_output",
    "call_id": "call_489952e5-9b3b-41af-90fb-1c3482858d06",
    "output": "\"2025-07-17\""
  }],
  "previous_response_id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOmdlbWluaTttb2RlbF9pZDo4NzU5YzhkMC02NGIxLTRkYTUtYWUxZS1mOWY1Nzg0MTU3NmM7cmVzcG9uc2VfaWQ6ZHFsNGFMdk1HTTI1cXNNUDVaQ1I0QWs=",
  "tools": [{
    "name": "get_date",
    "description": "Get current date in 'YYYY-MM-DD' format.\n",
    "parameters": {"type": "object", "properties": {}},
    "type": "function"
  }],
  "metadata": {"charge_multiplier": "1", "ai_feature": "date"}
}
```

### Result: ❌ Error

## Key Findings

1. **Session continuity works** ✅ - The encoded response IDs are working correctly
2. **Regular conversations work** ✅ - Non-tool messages work fine with `previous_response_id`
3. **Tool calling breaks** ❌ - Only when using `previous_response_id` with Gemini models
4. **OpenAI models work** ✅ - The same flow works correctly with OpenAI models

## Hypothesis

When reconstructing conversation context from `previous_response_id` for Gemini:
- The tool call history might not be properly restored
- Gemini expects the full conversation including the original function call
- The context reconstruction might be missing critical information about the function call turn

## Investigation Areas

1. **Context Reconstruction**: How does LiteLLM rebuild conversation history from `previous_response_id`?
2. **Gemini vs OpenAI**: What's different about how these providers handle tool call context?
3. **Message Format**: Does Gemini expect a different format for continuing tool conversations?
4. **State Management**: Is the function call state properly preserved in the encoded response ID?

## Impact

- Breaks tool/function calling workflows when using session continuity with Gemini
- Forces users to maintain full conversation history instead of using `previous_response_id`
- Creates inconsistency between providers (OpenAI works, Gemini doesn't)

## Related Information

- This issue was discovered after fixing streaming ID consistency (#PR_NUMBER)
- The streaming ID fix is working correctly - this is a separate issue
- Affects Gemini models specifically (tested with `gemini-2.0-flash`)

## Testing Results

### E2E Test Implementation

Created comprehensive end-to-end test in `test_session_continuity_tool_calling_e2e.py` that reproduces the issue:

**Test Setup:**
- LiteLLM proxy server running with both OpenAI and Gemini models
- Environment variables: `OPENAI_API_KEY` and `GEMINI_API_KEY`
- Tool definition: `get_date` function for testing
- Two-step flow: initial tool call → function result with `previous_response_id`

**Test Results (Confirmed):**

✅ **OpenAI (gpt-4o)**: Tool calling with session continuity works perfectly
- Initial request with tool → receives function call response
- Follow-up request with function result + `previous_response_id` → completes successfully
- Response IDs: 
  - Initial: `resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDpOb25lO3Jlc3BvbnNlX2lkOnJlc3BfNjg3OGIxMmMyYzY0ODE5YjhlNTkwNWNlNmFjMDQzNDQwMDk2MDU0OGQyYTgyN2Nl`
  - Follow-up: `resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDpOb25lO3Jlc3BvbnNlX2lkOnJlc3BfNjg3OGIxMmNmYWQ4ODE5YmJkZjg5MjQ0Y2MzZGZmYjMwMDk2MDU0OGQyYTgyN2Nl`

❌ **Gemini (gemini-2.5-flash)**: Tool calling with session continuity fails
- Initial request with tool → receives function call response successfully
- Follow-up request with function result + `previous_response_id` → **FAILS**
- Error: `litellm.BadRequestError: VertexAIException BadRequestError - Please ensure that function call turn comes immediately after a user turn or after a function response turn.`
- Initial response ID: `resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOmdlbWluaTttb2RlbF9pZDpOb25lO3Jlc3BvbnNlX2lkOkw3RjRhTTJXR3EtQWtkVVB6cU9WNEEw`

### Error Analysis

The Gemini error message reveals the core issue:
> "Please ensure that function call turn comes immediately after a user turn or after a function response turn."

This indicates that when LiteLLM reconstructs the conversation context from `previous_response_id`, **it's missing the original user turn** that initiated the function call. Gemini expects the conversation history to include:

1. User turn: "What date is it?" 
2. Assistant turn: Function call to `get_date`
3. Function response turn: "2025-07-17"  ← This is what we're providing
4. Assistant turn: Final response with the date

But the context reconstruction is only providing step 3, missing steps 1-2.

### Response ID Structure

Response IDs are base64-encoded and contain:
- Provider information (`custom_llm_provider:gemini` or `custom_llm_provider:openai`)
- Model ID 
- Original response ID

Example decoded: `litellm:custom_llm_provider:gemini;model_id:None;response_id:L7F4aM2WGq-AkdUPzqOV4A0`

## Next Steps

1. ✅ **Issue confirmed** - Bug reproduced successfully with E2E test
2. 🔍 **Investigate context reconstruction** - Find where `previous_response_id` is decoded and conversation history is rebuilt
3. 🔍 **Compare provider implementations** - Understand why OpenAI works but Gemini doesn't
4. 🛠️ **Fix context building** - Ensure Gemini gets complete conversation history including original user turn
5. ✅ **Validate fix** - Run E2E test to confirm both providers work