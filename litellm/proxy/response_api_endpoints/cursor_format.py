"""
Cursor Format Transformer

This module transforms OpenAI Chat Completions streaming format to Cursor's 
proprietary protobuf-JSON format for tool calls.

Cursor expects streaming responses in this format:
{
  "text": "...",
  "tool_call_v2": {
    "tool": <ClientSideToolV2 enum>,
    "tool_call_id": "...",
    "<tool>_params": {...}
  }
}

Instead of OpenAI's format:
{
  "choices": [{
    "delta": {
      "content": "...",
      "tool_calls": [{"id": "...", "function": {"name": "...", "arguments": "..."}}]
    }
  }]
}
"""

import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger


class ClientSideToolV2:
    """Cursor's ClientSideToolV2 enum values"""
    
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
    RIPGREP_RAW_SEARCH = 41
    GLOB_FILE_SEARCH = 42
    CALL_MCP_TOOL = 49
    ASK_QUESTION = 51
    COMPUTER_USE = 54
    WRITE_SHELL_STDIN = 55


# Map OpenAI function names to Cursor tool types and params keys
TOOL_NAME_MAP: Dict[str, Dict[str, Any]] = {
    # Standard names
    "read_file": {"tool": ClientSideToolV2.READ_FILE, "params_key": "read_file_params"},
    "edit_file": {"tool": ClientSideToolV2.EDIT_FILE, "params_key": "edit_file_params"},
    "str_replace": {"tool": ClientSideToolV2.EDIT_FILE_V2, "params_key": "edit_file_params"},
    "list_dir": {"tool": ClientSideToolV2.LIST_DIR, "params_key": "list_dir_params"},
    "delete_file": {"tool": ClientSideToolV2.DELETE_FILE, "params_key": "delete_file_params"},
    "run_terminal_command": {"tool": ClientSideToolV2.RUN_TERMINAL_COMMAND_V2, "params_key": "run_terminal_command_v2_params"},
    "shell": {"tool": ClientSideToolV2.RUN_TERMINAL_COMMAND_V2, "params_key": "run_terminal_command_v2_params"},
    "grep": {"tool": ClientSideToolV2.RIPGREP_SEARCH, "params_key": "ripgrep_search_params"},
    "glob": {"tool": ClientSideToolV2.GLOB_FILE_SEARCH, "params_key": "file_search_params"},
    "web_search": {"tool": ClientSideToolV2.WEB_SEARCH, "params_key": "web_search_params"},
    # LiteLLM / Claude tool names
    "Read": {"tool": ClientSideToolV2.READ_FILE_V2, "params_key": "read_file_params"},
    "Write": {"tool": ClientSideToolV2.EDIT_FILE_V2, "params_key": "edit_file_params"},
    "StrReplace": {"tool": ClientSideToolV2.EDIT_FILE_V2, "params_key": "edit_file_params"},
    "LS": {"tool": ClientSideToolV2.LIST_DIR_V2, "params_key": "list_dir_params"},
    "Shell": {"tool": ClientSideToolV2.RUN_TERMINAL_COMMAND_V2, "params_key": "run_terminal_command_v2_params"},
    "Delete": {"tool": ClientSideToolV2.DELETE_FILE, "params_key": "delete_file_params"},
    "Grep": {"tool": ClientSideToolV2.RIPGREP_RAW_SEARCH, "params_key": "ripgrep_search_params"},
    "Glob": {"tool": ClientSideToolV2.GLOB_FILE_SEARCH, "params_key": "file_search_params"},
    "TodoRead": {"tool": ClientSideToolV2.TODO_READ, "params_key": "todo_read_params"},
    "TodoWrite": {"tool": ClientSideToolV2.TODO_WRITE, "params_key": "todo_write_params"},
    "ReadLints": {"tool": ClientSideToolV2.SEMANTIC_SEARCH_FULL, "params_key": "semantic_search_full_params"},
    "EditNotebook": {"tool": ClientSideToolV2.EDIT_FILE_V2, "params_key": "edit_file_params"},
}


class CursorStreamChunk(BaseModel):
    """Represents a single streaming chunk in Cursor format"""
    
    text: str = ""
    tool_call_v2: Optional[Dict[str, Any]] = None
    partial_tool_call: Optional[Dict[str, Any]] = None
    thinking: Optional[Dict[str, Any]] = None


def transform_args_to_cursor_params(function_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform OpenAI function arguments to Cursor's parameter format.
    
    Args:
        function_name: The name of the function being called
        args: The function arguments from OpenAI format
        
    Returns:
        Cursor-format parameters dict
    """
    func_name_lower = function_name.lower()
    
    if func_name_lower in ["read_file", "read"]:
        return {
            "relative_workspace_path": args.get("path", args.get("file_path", "")),
            "read_entire_file": not (args.get("offset") or args.get("limit")),
            "start_line_one_indexed": args.get("offset"),
            "end_line_one_indexed_inclusive": (
                args.get("offset", 0) + args.get("limit") if args.get("limit") else None
            ),
            "max_lines": args.get("limit"),
        }
    elif func_name_lower in ["edit_file", "write", "strreplace", "str_replace"]:
        params: Dict[str, Any] = {
            "relative_workspace_path": args.get("path", args.get("file_path", "")),
        }
        if args.get("old_string") is not None:
            params["old_string"] = args.get("old_string", "")
            params["new_string"] = args.get("new_string", "")
        elif args.get("contents") is not None:
            params["contents"] = args.get("contents", "")
        if args.get("language"):
            params["language"] = args.get("language")
        return params
    elif func_name_lower in ["list_dir", "ls"]:
        return {
            "directory_path": args.get("path", args.get("target_directory", "."))
        }
    elif func_name_lower in ["run_terminal_command", "shell"]:
        return {
            "command": args.get("command", ""),
            "cwd": args.get("working_directory", args.get("cwd")),
            "is_background": args.get("is_background", False),
            "require_user_approval": args.get("require_user_approval", True),
        }
    elif func_name_lower in ["delete_file", "delete"]:
        return {
            "relative_workspace_path": args.get("path", "")
        }
    elif func_name_lower in ["grep"]:
        return {
            "query": args.get("pattern", args.get("query", "")),
            "case_sensitive": not args.get("-i", False),
            "file_pattern": args.get("glob", args.get("file_pattern")),
        }
    elif func_name_lower in ["glob"]:
        return {
            "glob_pattern": args.get("glob_pattern", args.get("pattern", "")),
            "directory_path": args.get("target_directory", "."),
        }
    elif func_name_lower in ["web_search"]:
        return {
            "query": args.get("query", ""),
        }
    elif func_name_lower in ["todoread"]:
        return {}
    elif func_name_lower in ["todowrite"]:
        return {
            "todos": args.get("todos", []),
            "merge": args.get("merge", False),
        }
    elif func_name_lower in ["editnotebook"]:
        return {
            "relative_workspace_path": args.get("target_notebook", ""),
            "notebook_cell_idx": args.get("cell_idx"),
            "is_new_cell": args.get("is_new_cell", False),
            "cell_language": args.get("cell_language"),
            "old_string": args.get("old_string", ""),
            "new_string": args.get("new_string", ""),
        }
    
    # Default: pass through for MCP or unknown tools
    return args


def openai_tool_call_to_cursor_format(tool_call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert an OpenAI tool_call to Cursor's StreamedBackToolCallV2 format.
    
    Args:
        tool_call: OpenAI format tool call dict with 'id', 'function.name', 'function.arguments'
        
    Returns:
        Cursor format tool_call_v2 dict, or None if conversion fails
    """
    try:
        function = tool_call.get("function", {})
        func_name = function.get("name", "")
        tool_call_id = tool_call.get("id", "")
        
        # Parse arguments
        arguments_str = function.get("arguments", "{}")
        try:
            args = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError:
            args = {}
        
        # Get tool mapping
        tool_info = TOOL_NAME_MAP.get(func_name)
        if not tool_info:
            # Check case-insensitive
            for name, info in TOOL_NAME_MAP.items():
                if name.lower() == func_name.lower():
                    tool_info = info
                    break
        
        # Default to MCP for unknown tools
        if not tool_info:
            tool_info = {"tool": ClientSideToolV2.MCP, "params_key": "mcp_params"}
        
        # Transform arguments to Cursor params
        cursor_params = transform_args_to_cursor_params(func_name, args)
        
        # Build Cursor format
        result: Dict[str, Any] = {
            "tool": tool_info["tool"],
            "tool_call_id": tool_call_id,
        }
        result[tool_info["params_key"]] = cursor_params
        
        return result
        
    except Exception as e:
        verbose_proxy_logger.error(f"Error converting tool call to Cursor format: {e}")
        return None


def openai_chunk_to_cursor_format(openai_chunk: Union[Dict[str, Any], BaseModel]) -> Dict[str, Any]:
    """
    Convert an OpenAI streaming chunk to Cursor's streaming format.
    
    Args:
        openai_chunk: OpenAI ModelResponseStream or dict with choices[].delta
        
    Returns:
        Cursor format dict with 'text', 'tool_call_v2', etc.
    """
    # Convert BaseModel to dict if needed
    if isinstance(openai_chunk, BaseModel):
        chunk_dict = openai_chunk.model_dump(exclude_none=True, exclude_unset=True)
    else:
        chunk_dict = openai_chunk
    
    cursor_chunk: Dict[str, Any] = {"text": ""}
    
    # Extract from choices
    choices = chunk_dict.get("choices", [])
    if not choices:
        return cursor_chunk
    
    delta = choices[0].get("delta", {})
    
    # Handle text content
    content = delta.get("content")
    if content:
        cursor_chunk["text"] = content
    
    # Handle reasoning/thinking content
    reasoning = delta.get("reasoning_content")
    thinking_blocks = delta.get("thinking_blocks")
    if reasoning or thinking_blocks:
        cursor_chunk["thinking"] = {
            "thinking": reasoning or "",
            "signature": "",
        }
        if thinking_blocks and len(thinking_blocks) > 0:
            cursor_chunk["thinking"]["thinking"] = thinking_blocks[0].get("thinking", "")
    
    # Handle tool calls
    tool_calls = delta.get("tool_calls", [])
    if tool_calls:
        tc = tool_calls[0]
        function = tc.get("function", {})
        func_name = function.get("name", "")
        
        # If we have arguments, it's a full tool call
        if function.get("arguments"):
            cursor_tool_call = openai_tool_call_to_cursor_format(tc)
            if cursor_tool_call:
                cursor_chunk["tool_call_v2"] = cursor_tool_call
        # If we only have name, it's a partial tool call announcement
        elif func_name:
            tool_info = TOOL_NAME_MAP.get(func_name)
            if not tool_info:
                for name, info in TOOL_NAME_MAP.items():
                    if name.lower() == func_name.lower():
                        tool_info = info
                        break
            if not tool_info:
                tool_info = {"tool": ClientSideToolV2.MCP, "params_key": "mcp_params"}
            
            cursor_chunk["partial_tool_call"] = {
                "tool": tool_info["tool"],
                "tool_call_id": tc.get("id", ""),
                "name": func_name,
                "tool_index": tc.get("index", 0),
            }
    
    return cursor_chunk


async def cursor_async_data_generator(
    response,
    user_api_key_dict,
    request_data: dict,
    proxy_logging_obj,
):
    """
    Async generator that yields streaming chunks in Cursor format.
    
    This replaces async_data_generator for the /cursor/chat/completions endpoint.
    It converts OpenAI format chunks to Cursor's protobuf-JSON format.
    
    Args:
        response: The streaming response iterator
        user_api_key_dict: User API key authentication dict
        request_data: Request data dict
        proxy_logging_obj: Proxy logging object
        
    Yields:
        SSE-formatted strings in Cursor's format
    """
    verbose_proxy_logger.debug("inside cursor_async_data_generator")
    
    try:
        # Track accumulated tool call arguments for streaming
        tool_call_buffer: Dict[str, Dict[str, Any]] = {}  # tool_call_id -> accumulated data
        
        async for chunk in proxy_logging_obj.async_post_call_streaming_iterator_hook(
            user_api_key_dict=user_api_key_dict,
            response=response,
            request_data=request_data,
        ):
            verbose_proxy_logger.debug(
                f"cursor_async_data_generator: received streaming chunk - {chunk}"
            )
            
            # Apply post-call hooks
            chunk = await proxy_logging_obj.async_post_call_streaming_hook(
                user_api_key_dict=user_api_key_dict,
                response=chunk,
                data=request_data,
                str_so_far="",
            )
            
            # Convert to Cursor format
            if isinstance(chunk, BaseModel):
                cursor_chunk = openai_chunk_to_cursor_format(chunk)
            elif isinstance(chunk, dict):
                cursor_chunk = openai_chunk_to_cursor_format(chunk)
            elif isinstance(chunk, str):
                # Already a string, check if it's an error
                if chunk.startswith("data: "):
                    yield chunk
                    continue
                cursor_chunk = {"text": chunk}
            else:
                continue
            
            # Handle streaming tool call argument accumulation
            # OpenAI streams arguments in multiple chunks, we need to accumulate
            if isinstance(chunk, BaseModel):
                chunk_dict = chunk.model_dump(exclude_none=True, exclude_unset=True)
            else:
                chunk_dict = chunk if isinstance(chunk, dict) else {}
            
            choices = chunk_dict.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                tool_calls = delta.get("tool_calls", [])
                if tool_calls:
                    tc = tool_calls[0]
                    tc_id = tc.get("id")
                    function = tc.get("function", {})
                    
                    if tc_id:
                        # Initialize buffer for this tool call
                        if tc_id not in tool_call_buffer:
                            tool_call_buffer[tc_id] = {
                                "id": tc_id,
                                "function": {"name": "", "arguments": ""},
                                "type": "function",
                                "index": tc.get("index", 0),
                            }
                        
                        # Accumulate name
                        if function.get("name"):
                            tool_call_buffer[tc_id]["function"]["name"] = function["name"]
                        
                        # Accumulate arguments
                        if function.get("arguments"):
                            tool_call_buffer[tc_id]["function"]["arguments"] += function["arguments"]
                            
                            # Try to parse arguments - if valid JSON, emit the full tool call
                            try:
                                json.loads(tool_call_buffer[tc_id]["function"]["arguments"])
                                # Valid JSON - emit full tool call
                                cursor_tool_call = openai_tool_call_to_cursor_format(
                                    tool_call_buffer[tc_id]
                                )
                                if cursor_tool_call:
                                    cursor_chunk = {"text": "", "tool_call_v2": cursor_tool_call}
                                    # Clear buffer
                                    del tool_call_buffer[tc_id]
                            except json.JSONDecodeError:
                                # Not complete yet, emit partial if we have name
                                if tool_call_buffer[tc_id]["function"]["name"]:
                                    func_name = tool_call_buffer[tc_id]["function"]["name"]
                                    tool_info = TOOL_NAME_MAP.get(func_name)
                                    if not tool_info:
                                        for name, info in TOOL_NAME_MAP.items():
                                            if name.lower() == func_name.lower():
                                                tool_info = info
                                                break
                                    if not tool_info:
                                        tool_info = {"tool": ClientSideToolV2.MCP}
                                    
                                    cursor_chunk = {
                                        "text": "",
                                        "partial_tool_call": {
                                            "tool": tool_info["tool"],
                                            "tool_call_id": tc_id,
                                            "name": func_name,
                                            "tool_index": tc.get("index", 0),
                                        }
                                    }
                                else:
                                    # Skip this chunk, waiting for more data
                                    continue
            
            # Yield the Cursor-formatted chunk
            chunk_json = json.dumps(cursor_chunk)
            yield f"data: {chunk_json}\n\n"
        
        # Emit any remaining buffered tool calls
        for tc_id, tc_data in tool_call_buffer.items():
            if tc_data["function"]["arguments"]:
                cursor_tool_call = openai_tool_call_to_cursor_format(tc_data)
                if cursor_tool_call:
                    cursor_chunk = {"text": "", "tool_call_v2": cursor_tool_call}
                    chunk_json = json.dumps(cursor_chunk)
                    yield f"data: {chunk_json}\n\n"
        
        # Done
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        verbose_proxy_logger.exception(
            f"cursor_async_data_generator: Exception occurred - {str(e)}"
        )
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
