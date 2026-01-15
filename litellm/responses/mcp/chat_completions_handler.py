"""Helpers for handling MCP-aware `/chat/completions` requests."""

from typing import (
    Any,
    List,
    Optional,
    Union,
)

from litellm.responses.mcp.litellm_proxy_mcp_handler import (
    LiteLLM_Proxy_MCP_Handler,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper


async def acompletion_with_mcp(
    model: str,
    messages: List,
    tools: Optional[List] = None,
    **kwargs: Any,
) -> Union[ModelResponse, CustomStreamWrapper]:
    """
    Async completion with MCP integration.

    This function handles MCP tool integration following the same pattern as aresponses_api_with_mcp.
    It's designed to be called from the synchronous completion() function and return a coroutine.

    When MCP tools with server_url="litellm_proxy" are provided, this function will:
    1. Get available tools from the MCP server manager
    2. Transform them to OpenAI format
    3. Call acompletion with the transformed tools
    4. If require_approval="never" and tool calls are returned, automatically execute them
    5. Make a follow-up call with the tool results
    """
    from litellm import acompletion as litellm_acompletion

    # Parse MCP tools and separate from other tools
    (
        mcp_tools_with_litellm_proxy,
        other_tools,
    ) = LiteLLM_Proxy_MCP_Handler._parse_mcp_tools(tools)

    if not mcp_tools_with_litellm_proxy:
        # No MCP tools, proceed with regular completion
        return await litellm_acompletion(
            model=model,
            messages=messages,
            tools=tools,
            **kwargs,
        )

    # Extract user_api_key_auth from metadata or kwargs
    user_api_key_auth = kwargs.get("user_api_key_auth") or (
        (kwargs.get("metadata", {}) or {}).get("user_api_key_auth")
    )

    # Process MCP tools
    (
        deduplicated_mcp_tools,
        tool_server_map,
    ) = await LiteLLM_Proxy_MCP_Handler._process_mcp_tools_without_openai_transform(
        user_api_key_auth=user_api_key_auth,
        mcp_tools_with_litellm_proxy=mcp_tools_with_litellm_proxy,
    )

    openai_tools = LiteLLM_Proxy_MCP_Handler._transform_mcp_tools_to_openai(
        deduplicated_mcp_tools,
        target_format="chat",
    )

    # Combine with other tools
    all_tools = openai_tools + other_tools if (openai_tools or other_tools) else None

    # Determine if we should auto-execute tools
    should_auto_execute = LiteLLM_Proxy_MCP_Handler._should_auto_execute_tools(
        mcp_tools_with_litellm_proxy=mcp_tools_with_litellm_proxy
    )

    # Extract MCP auth headers
    (
        mcp_auth_header,
        mcp_server_auth_headers,
        oauth2_headers,
        raw_headers,
    ) = ResponsesAPIRequestUtils.extract_mcp_headers_from_request(
        secret_fields=kwargs.get("secret_fields"),
        tools=tools,
    )

    # Prepare call parameters
    # Remove keys that shouldn't be passed to acompletion
    clean_kwargs = {k: v for k, v in kwargs.items() if k not in ["acompletion"]}

    base_call_args = {
        "model": model,
        "messages": messages,
        "tools": all_tools,
        "_skip_mcp_handler": True,  # Prevent recursion
        **clean_kwargs,
    }

    # If not auto-executing, just make the call with transformed tools
    if not should_auto_execute:
        return await litellm_acompletion(**base_call_args)

    # For auto-execute: disable streaming for initial call
    stream = kwargs.get("stream", False)
    mock_tool_calls = base_call_args.pop("mock_tool_calls", None)

    initial_call_args = dict(base_call_args)
    initial_call_args["stream"] = False
    if mock_tool_calls is not None:
        initial_call_args["mock_tool_calls"] = mock_tool_calls

    # Make initial call
    initial_response = await litellm_acompletion(**initial_call_args)

    if not isinstance(initial_response, ModelResponse):
        return initial_response

    # Extract tool calls from response
    tool_calls = LiteLLM_Proxy_MCP_Handler._extract_tool_calls_from_chat_response(
        response=initial_response
    )

    if not tool_calls:
        # No tool calls, return response or retry with streaming if needed
        if stream:
            retry_args = dict(base_call_args)
            retry_args["stream"] = stream
            return await litellm_acompletion(**retry_args)
        return initial_response

    # Execute tool calls
    tool_results = await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map=tool_server_map,
        tool_calls=tool_calls,
        user_api_key_auth=user_api_key_auth,
        mcp_auth_header=mcp_auth_header,
        mcp_server_auth_headers=mcp_server_auth_headers,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
    )

    if not tool_results:
        return initial_response

    # Create follow-up messages with tool results
    follow_up_messages = LiteLLM_Proxy_MCP_Handler._create_follow_up_messages_for_chat(
        original_messages=messages,
        response=initial_response,
        tool_results=tool_results,
    )

    # Make follow-up call with original stream setting
    follow_up_call_args = dict(base_call_args)
    follow_up_call_args["messages"] = follow_up_messages
    follow_up_call_args["stream"] = stream

    return await litellm_acompletion(**follow_up_call_args)
