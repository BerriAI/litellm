# Cursor BYOK Tool Calls Analysis

## Key Discovery: Server-Side Proxying

After deep analysis of the Cursor client code, I found that **Cursor's server proxies BYOK requests**. This is critical!

### How BYOK Works (Simplified Flow)

```
┌──────────────┐      ┌────────────────────┐      ┌─────────────────────┐
│ Cursor IDE   │──────▶ Cursor Server      │──────▶ User's BYOK Endpoint│
│ (Client)     │◀──────│ (api2.cursor.sh)   │◀──────│ (LiteLLM)           │
└──────────────┘      └────────────────────┘      └─────────────────────┘
    Protobuf              Proxy Request           OpenAI JSON
    Response              with ModelDetails       Request/Response
```

### What I Found in the Code

1. **ModelDetails Protobuf** contains `openai_api_base_url`:
```javascript
// From workbench.desktop.main.js
jg = class { // ModelDetails
  fields = [
    {no:4, name:"openai_api_base_url", kind:"scalar", T:9, opt:true},
    {no:1, name:"api_key", kind:"scalar", T:9, opt:true},
    {no:2, name:"model_name", kind:"scalar", T:9},
    // ... more fields
  ]
}
```

2. **Client sends ModelDetails to server**, server makes the actual API call:
```javascript
getModelDetailsFromName(modelName, maxMode) {
  return new jg({
    apiKey: userApiKey,
    modelName: serverModelName,
    openaiApiBaseUrl: this._reactiveStorageService.applicationUserPersistentStorage.openAIBaseUrl,
    // ...
  })
}
```

3. **Server returns StreamUnifiedChatResponse** which includes tool_call fields:
```javascript
// StreamUnifiedChatResponse has these oneof options:
// - text
// - tool_call_v2 
// - partial_tool_call
// - thinking
// etc.
```

## The Opportunity

Since Cursor's **server** (not client) handles the actual BYOK API call:

1. The server receives OpenAI JSON response including `tool_calls`
2. The server should be able to convert `tool_calls` to its protobuf `tool_call_v2` format
3. The server sends protobuf back to client
4. Client executes tools as normal

**The conversion logic exists on the server!** The issue may be that it's not enabled for BYOK.

## Proposed Solution for LiteLLM

Since we can't modify Cursor's server, we should output in **Cursor's protobuf-JSON format** if Cursor's server expects JSON that it can parse into protobuf.

### Option 1: Output Cursor's Native JSON Format

Transform OpenAI tool_calls to Cursor's expected JSON format:

```python
# OpenAI format
{
  "choices": [{
    "delta": {
      "tool_calls": [{
        "id": "call_123",
        "function": {
          "name": "read_file",
          "arguments": "{\"path\": \"foo.py\"}"
        }
      }]
    }
  }]
}

# Cursor's expected format (based on protobuf)
{
  "tool_call_v2": {
    "tool": 5,  # READ_FILE enum value
    "tool_call_id": "call_123",
    "read_file": {
      "paths": ["foo.py"]
    }
  }
}
```

### Option 2: Return Both Formats

Stream both OpenAI format AND Cursor's format in case either works:
- First send Cursor's `tool_call_v2` format
- Then send standard OpenAI `tool_calls` format

### Tool Enum Mapping

From exec-daemon analysis:
```javascript
ClientSideToolV2 = {
  UNSPECIFIED_TOOL: 0,
  SEARCH_CODEBASE: 1,
  SEARCH_WEB: 2,
  URL: 3,
  ATTEMPT_COMPLETION: 4,
  READ_FILE: 5,
  LIST_DIR: 6,
  EDIT_FILE: 7,
  FILE_SEARCH: 8,
  RUN_TERMINAL_COMMAND: 17,
  MCP: 19,
  // ... more
}
```

## Next Steps

1. **Test Hypothesis**: Modify `/cursor/chat/completions` to output Cursor's JSON format
2. **Check if Cursor parses it**: See if tools execute
3. **If not working**: May need to investigate Cursor's server-side code

## Files Modified

- `/workspace/litellm/proxy/response_api_endpoints/endpoints.py` - Main endpoint (UPDATED)
- `/workspace/litellm/proxy/response_api_endpoints/cursor_format.py` - Transformation logic

## Changes Made

### 1. Updated `/cursor/chat/completions` endpoint

The endpoint now outputs Cursor's native streaming format instead of OpenAI format:

```python
# Before: OpenAI format
{"choices": [{"delta": {"content": "Hello", "tool_calls": [...]}}]}

# After: Cursor's native format  
{"text": "Hello", "tool_call_v2": {"tool": 5, "read_file_params": {...}}}
```

### 2. Tool Call Transformation

The `cursor_format.py` module transforms OpenAI tool_calls to Cursor's format:

- Maps function names to `ClientSideToolV2` enum values
- Transforms arguments to Cursor's parameter format
- Handles streaming tool call accumulation (arguments arrive in chunks)

### 3. Supported Tools

| OpenAI Function | Cursor Tool | Enum Value |
|-----------------|-------------|------------|
| `read_file`, `Read` | READ_FILE | 5 |
| `edit_file`, `Write`, `StrReplace` | EDIT_FILE_V2 | 38 |
| `list_dir`, `LS` | LIST_DIR_V2 | 39 |
| `run_terminal_command`, `Shell` | RUN_TERMINAL_COMMAND_V2 | 15 |
| `delete_file`, `Delete` | DELETE_FILE | 11 |
| `grep`, `Grep` | RIPGREP_SEARCH | 3/41 |
| `glob`, `Glob` | GLOB_FILE_SEARCH | 42 |
| `web_search` | WEB_SEARCH | 18 |
| Unknown tools | MCP | 19 |

## Testing

To test if Cursor picks up tool calls:

1. Configure Cursor to use your LiteLLM endpoint as `openai_api_base_url`
2. Start a chat that triggers tool calls (e.g., "read the file foo.py")
3. Check if Cursor's client executes the tools

## Questions Remaining

1. Does Cursor's client parse this JSON format from BYOK responses?
2. If not, is the server supposed to transform it before sending to client?
3. May need to check Cursor's server-side BYOK handling code
