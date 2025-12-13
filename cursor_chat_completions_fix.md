# What to Change in `/cursor/chat/completions` for Tool Calls

## Status: IMPLEMENTED ✅

The changes have been implemented in:
- **NEW FILE**: `litellm/proxy/response_api_endpoints/cursor_format.py` - Cursor format transformer
- **UPDATED**: `litellm/proxy/response_api_endpoints/endpoints.py` - Uses new Cursor format generator

## The Core Problem

Cursor expects tool calls in its **proprietary protobuf format**, not standard OpenAI `tool_calls` format. Even though protobuf messages have `fromJson` methods, Cursor's BYOK client may not be parsing them correctly.

## Cursor's Expected Streaming Response Format

When Cursor receives a streaming response, it expects chunks in this JSON structure (protobuf-to-JSON):

```json
{
  "text": "",
  "tool_call_v2": {
    "tool": 5,  // ClientSideToolV2 enum: READ_FILE=5, EDIT_FILE=7, etc.
    "tool_call_id": "call_abc123",
    "read_file_params": {
      "relative_workspace_path": "src/app.py",
      "read_entire_file": true
    }
  }
}
```

NOT the OpenAI format:
```json
{
  "choices": [{
    "delta": {
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {"name": "read_file", "arguments": "{}"}
      }]
    }
  }]
}
```

## ClientSideToolV2 Enum Values

```
UNSPECIFIED = 0
READ_SEMSEARCH_FILES = 1
RIPGREP_SEARCH = 3
READ_FILE = 5
LIST_DIR = 6
EDIT_FILE = 7
FILE_SEARCH = 8
SEMANTIC_SEARCH_FULL = 9
DELETE_FILE = 11
REAPPLY = 12
RUN_TERMINAL_COMMAND_V2 = 15
FETCH_RULES = 16
WEB_SEARCH = 18
MCP = 19
SEARCH_SYMBOLS = 23
GO_TO_DEFINITION = 31
TASK = 32
TODO_READ = 34
TODO_WRITE = 35
EDIT_FILE_V2 = 38
LIST_DIR_V2 = 39
READ_FILE_V2 = 40
GLOB_FILE_SEARCH = 42
CALL_MCP_TOOL = 49
ASK_QUESTION = 51
COMPUTER_USE = 54
```

## Tool Params Structures (JSON)

### ReadFileParams (tool=5)
```json
{
  "relative_workspace_path": "string",
  "read_entire_file": true,
  "start_line_one_indexed": 1,       // optional
  "end_line_one_indexed_inclusive": 100,  // optional
  "max_lines": 500,                   // optional
  "max_chars": 10000                  // optional
}
```

### EditFileParams (tool=7)
```json
{
  "relative_workspace_path": "string",
  "language": "python",
  "blocking": true,
  "contents": "full file contents",    // for write
  "old_string": "text to find",        // for search-replace
  "new_string": "replacement text",    // for search-replace
  "instructions": "optional instructions"
}
```

### ListDirParams (tool=6)
```json
{
  "directory_path": "string"
}
```

### RunTerminalCommandV2Params (tool=15)
```json
{
  "command": "npm install",
  "cwd": "/workspace",               // optional
  "is_background": false,
  "require_user_approval": true
}
```

### DeleteFileParams (tool=11)
```json
{
  "relative_workspace_path": "string"
}
```

## Required Changes to `/cursor/chat/completions`

### Option 1: Transform to Cursor's Format (Recommended)

Modify the streaming response to output Cursor's format instead of OpenAI format:

```python
# In litellm/proxy/response_api_endpoints/endpoints.py

def transform_tool_call_to_cursor_format(openai_tool_call):
    """Transform OpenAI tool_call to Cursor's StreamedBackToolCallV2 format"""
    
    function_name = openai_tool_call.get("function", {}).get("name", "")
    arguments = json.loads(openai_tool_call.get("function", {}).get("arguments", "{}"))
    tool_call_id = openai_tool_call.get("id", "")
    
    # Map function names to Cursor tool types
    TOOL_NAME_MAP = {
        "read_file": {"tool": 5, "params_key": "read_file_params"},
        "Read": {"tool": 40, "params_key": "read_file_params"},  # READ_FILE_V2
        "edit_file": {"tool": 7, "params_key": "edit_file_params"},
        "StrReplace": {"tool": 38, "params_key": "edit_file_params"},  # EDIT_FILE_V2
        "Write": {"tool": 38, "params_key": "edit_file_params"},
        "list_dir": {"tool": 6, "params_key": "list_dir_params"},
        "LS": {"tool": 39, "params_key": "list_dir_params"},  # LIST_DIR_V2
        "run_terminal_command": {"tool": 15, "params_key": "run_terminal_command_v2_params"},
        "Shell": {"tool": 15, "params_key": "run_terminal_command_v2_params"},
        "delete_file": {"tool": 11, "params_key": "delete_file_params"},
        "Delete": {"tool": 11, "params_key": "delete_file_params"},
        "grep": {"tool": 3, "params_key": "ripgrep_search_params"},
        "Grep": {"tool": 41, "params_key": "ripgrep_search_params"},  # RIPGREP_RAW_SEARCH
        "glob": {"tool": 42, "params_key": "file_search_params"},
        "Glob": {"tool": 42, "params_key": "file_search_params"},
    }
    
    tool_info = TOOL_NAME_MAP.get(function_name, {"tool": 19, "params_key": "mcp_params"})
    
    # Transform arguments to Cursor's param format
    cursor_params = transform_args_to_cursor_params(function_name, arguments)
    
    return {
        "tool": tool_info["tool"],
        "tool_call_id": tool_call_id,
        tool_info["params_key"]: cursor_params
    }

def transform_args_to_cursor_params(function_name, args):
    """Transform OpenAI function arguments to Cursor's params format"""
    
    if function_name in ["read_file", "Read"]:
        return {
            "relative_workspace_path": args.get("path", args.get("file_path", "")),
            "read_entire_file": True,
            "start_line_one_indexed": args.get("offset"),
            "end_line_one_indexed_inclusive": args.get("limit")
        }
    elif function_name in ["edit_file", "StrReplace", "Write"]:
        return {
            "relative_workspace_path": args.get("path", args.get("file_path", "")),
            "old_string": args.get("old_string", ""),
            "new_string": args.get("new_string", args.get("contents", "")),
            "language": args.get("language", "")
        }
    elif function_name in ["list_dir", "LS"]:
        return {
            "directory_path": args.get("path", args.get("target_directory", ""))
        }
    elif function_name in ["run_terminal_command", "Shell"]:
        return {
            "command": args.get("command", ""),
            "cwd": args.get("working_directory"),
            "is_background": args.get("is_background", False),
            "require_user_approval": True
        }
    elif function_name in ["delete_file", "Delete"]:
        return {
            "relative_workspace_path": args.get("path", "")
        }
    # Default: pass through as-is for MCP
    return args
```

### Option 2: Output Cursor's Streaming Format

Change the SSE output format from:
```
data: {"choices":[{"delta":{"tool_calls":[...]}}]}
```

To Cursor's format:
```
data: {"text":"","tool_call_v2":{"tool":5,"tool_call_id":"...","read_file_params":{...}}}
```

### Streaming Sequence for Tool Calls

Cursor expects this sequence:

1. **Partial tool call** (signals tool call starting):
```json
{"partial_tool_call": {"tool": 5, "tool_call_id": "call_123", "name": "read_file", "tool_index": 0}}
```

2. **Full tool call** (with params):
```json
{"tool_call_v2": {"tool": 5, "tool_call_id": "call_123", "read_file_params": {...}}}
```

3. **Text continues** (after tool result):
```json
{"text": "Based on the file contents..."}
```

## Implementation Steps

1. **Create tool name mapping**: Map OpenAI function names → Cursor ClientSideToolV2 enum
2. **Create params transformer**: Convert OpenAI arguments → Cursor params structure
3. **Modify streaming generator**: Output Cursor format instead of OpenAI format
4. **Handle partial tool calls**: Emit `partial_tool_call` before `tool_call_v2`
5. **Test with Cursor**: Verify tool calls are executed

## Example Full Implementation

```python
# cursor_format_transformer.py

import json
from typing import Any, Dict, Optional

CURSOR_TOOL_MAP = {
    # Standard tool names
    "read_file": 5,
    "edit_file": 7, 
    "list_dir": 6,
    "delete_file": 11,
    "run_terminal_command": 15,
    "grep": 3,
    "glob": 42,
    "web_search": 18,
    # LiteLLM tool names
    "Read": 40,
    "Write": 38,
    "StrReplace": 38,
    "LS": 39,
    "Shell": 15,
    "Delete": 11,
    "Grep": 41,
    "Glob": 42,
}

def openai_chunk_to_cursor_format(openai_chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenAI streaming chunk to Cursor format"""
    
    cursor_chunk = {"text": ""}
    
    # Handle text content
    choices = openai_chunk.get("choices", [])
    if choices:
        delta = choices[0].get("delta", {})
        content = delta.get("content")
        if content:
            cursor_chunk["text"] = content
        
        # Handle tool calls
        tool_calls = delta.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]
            func = tc.get("function", {})
            func_name = func.get("name", "")
            
            if func_name:
                tool_enum = CURSOR_TOOL_MAP.get(func_name, 19)  # Default to MCP
                args = json.loads(func.get("arguments", "{}")) if func.get("arguments") else {}
                
                cursor_chunk["tool_call_v2"] = {
                    "tool": tool_enum,
                    "tool_call_id": tc.get("id", ""),
                    **get_cursor_params(func_name, args)
                }
    
    return cursor_chunk

def get_cursor_params(func_name: str, args: Dict) -> Dict[str, Any]:
    """Get Cursor-format params based on function name"""
    
    if func_name in ["read_file", "Read"]:
        return {"read_file_params": {
            "relative_workspace_path": args.get("path", ""),
            "read_entire_file": True
        }}
    elif func_name in ["edit_file", "Write", "StrReplace"]:
        return {"edit_file_params": {
            "relative_workspace_path": args.get("path", ""),
            "old_string": args.get("old_string", ""),
            "new_string": args.get("new_string", args.get("contents", ""))
        }}
    elif func_name in ["list_dir", "LS"]:
        return {"list_dir_params": {
            "directory_path": args.get("path", args.get("target_directory", ""))
        }}
    elif func_name in ["run_terminal_command", "Shell"]:
        return {"run_terminal_command_v2_params": {
            "command": args.get("command", ""),
            "is_background": args.get("is_background", False),
            "require_user_approval": True
        }}
    elif func_name in ["delete_file", "Delete"]:
        return {"delete_file_params": {
            "relative_workspace_path": args.get("path", "")
        }}
    else:
        # Generic MCP tool
        return {"mcp_params": {"tools": [{"name": func_name, "parameters": json.dumps(args)}]}}
```

## Key Insight

The fix requires outputting responses in **Cursor's protobuf-JSON format**, not OpenAI's Chat Completions format. Cursor's client expects specific field names like `tool_call_v2`, `read_file_params`, etc.

## Testing

After implementing, test with:
1. Simple text response (should work already)
2. Tool call that reads a file
3. Tool call that edits a file
4. Tool call that runs a command

Watch Cursor's behavior - if it shows "Running tool..." and executes, it's working!
