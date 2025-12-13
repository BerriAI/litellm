# Cursor API Tool Calls Analysis

## Key Finding: Cursor Uses Protobuf, Not OpenAI JSON Format

When Cursor communicates with its backend (`api2.cursor.sh`), it uses **Protocol Buffers over gRPC** with its own message types. Tool calls are NOT in OpenAI Chat Completions format - they're in Cursor's proprietary format.

## Cursor's Internal Tool Call Format

### Streaming Response Tool Calls (`aiserver.v1.StreamedBackToolCall`)

```protobuf
message StreamedBackToolCall {
  ClientSideToolV2 tool = 1;    // Enum: READ_FILE, EDIT_FILE, etc.
  string tool_call_id = 2;
  
  // Tool-specific params (oneof)
  oneof params {
    ReadFileStream read_file_stream = 7;
    EditFileStream edit_file_stream = 13;
    ListDirStream list_dir_stream = 12;
    RunTerminalCommandV2Stream run_terminal_command_v2_stream = 25;
    // ... 50+ other tool types
  }
}
```

### Agent Interaction Updates (`agent.v1.InteractionUpdate`)

For agent mode, Cursor uses a different message type:

```protobuf
message InteractionUpdate {
  oneof message {
    TextDeltaUpdate text_delta = 1;
    PartialToolCallUpdate partial_tool_call = 7;
    ToolCallDeltaUpdate tool_call_delta = 15;
    ToolCallStartedUpdate tool_call_started = 2;
    ToolCallCompletedUpdate tool_call_completed = 3;
    ThinkingDeltaUpdate thinking_delta = 4;
    // ... other updates
  }
}
```

## What This Means for BYOK/Custom Endpoints

When using BYOK with a custom `openai_api_base_url`:

1. **Cursor sends**: Responses API format requests (with `input` field) via HTTP/JSON
2. **Cursor expects**: OpenAI-compatible streaming responses

However, **Cursor's BYOK mode may have limited tool call support**. The exec-daemon code shows:
- Tool calls are handled internally through Cursor's proprietary protobuf format
- The `openai_api_base_url` is just passed to `ModelDetails` for routing
- There's no visible code parsing OpenAI-format `tool_calls` from BYOK responses

## Hypothesis: BYOK Doesn't Support Tool Calls

Based on the code analysis, **Cursor's BYOK mode likely does NOT support tool calls from custom endpoints**. When using BYOK:

1. Text streaming works (via standard SSE `data: {...}\n\n` format)
2. Tool calls are ignored because Cursor only processes tool calls through its native protobuf format
3. This explains why tool calls "don't take any action" - Cursor isn't parsing them

## LiteLLM's `/cursor/chat/completions` Response Format

LiteLLM transforms Responses API → Chat Completions format:

```json
// Streaming chunk with tool call
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "choices": [{
    "index": 0,
    "delta": {
      "tool_calls": [{
        "index": 0,
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "read_file",
          "arguments": "{\"path\": \"/src/app.py\"}"
        }
      }]
    },
    "finish_reason": null
  }]
}
```

This is valid OpenAI format, but Cursor may not be processing it when using BYOK.

## Recommendations

### Option 1: Confirm with Cursor Documentation
Check if Cursor officially supports tool calls when using BYOK/custom base URL. It may be a premium/enterprise feature.

### Option 2: Use Cursor's Native Format
If LiteLLM wants to support Cursor tool calls fully, it would need to:
1. Accept requests in Cursor's protobuf format
2. Return responses in Cursor's protobuf format
3. This is complex and may violate Cursor's ToS

### Option 3: Request Feature from Cursor
Ask Cursor to add tool call support for BYOK mode using standard OpenAI format.

## Testing Approach

To verify if Cursor ignores tool calls from BYOK:

1. Set up LiteLLM proxy with `/cursor/chat/completions`
2. Configure Cursor to use it as BYOK endpoint
3. Send a request that triggers a tool call
4. Check if:
   - Response streams correctly (text should work)
   - Tool calls appear in the response JSON
   - Cursor takes action on the tool calls (likely not)

## Verified Facts from Code Analysis

1. ✅ Cursor uses `api2.cursor.sh` as its API endpoint
2. ✅ Cursor uses protobuf (`aiserver.v1.*` messages) for communication
3. ✅ Tool calls use `StreamedBackToolCall` and `ClientSideToolV2` enum
4. ✅ BYOK uses `openai_api_base_url` in `ModelDetails`
5. ❓ Whether BYOK responses with tool calls are parsed is unclear
6. ❓ The exec-daemon doesn't show OpenAI tool call parsing code
