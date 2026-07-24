"""
MCP gateway support for the Anthropic `/v1/messages` API.

Mirrors ``litellm.responses.mcp.chat_completions_handler`` but speaks the
Anthropic Messages shapes: tools carry an ``input_schema``, the model asks for a
tool through a ``tool_use`` content block, and results are fed back as
``tool_result`` blocks in a user message.
"""

from typing import Any, AsyncIterator, Mapping, Sequence, Union

from litellm._logging import verbose_logger
from litellm.responses.mcp.request_context import MCPRequestContext
from litellm.types.llms.anthropic import (
    AnthropicMessagesTool,
    AnthropicMessagesToolResultParam,
    AnthropicMessagesUserMessageParam,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)

MAX_MCP_TOOL_USE_ITERATIONS = 10


def _get_response_content(response: AnthropicMessagesResponse) -> Sequence[Mapping[str, Any]]:
    content = response.get("content")
    if not isinstance(content, list):
        return ()
    return tuple(block for block in content if isinstance(block, dict))


def _extract_tool_use_blocks(response: AnthropicMessagesResponse) -> Sequence[Mapping[str, Any]]:
    """Return the ``tool_use`` content blocks the model emitted."""
    return tuple(block for block in _get_response_content(response) if block.get("type") == "tool_use")


def _get_stop_reason(response: AnthropicMessagesResponse) -> Union[str, None]:
    stop_reason = response.get("stop_reason")
    return stop_reason if isinstance(stop_reason, str) else None


def _build_tool_result_message(tool_results: Sequence[Mapping[str, Any]]) -> AnthropicMessagesUserMessageParam:
    """Turn executed tool results into the user message Anthropic expects."""
    return AnthropicMessagesUserMessageParam(
        role="user",
        content=tuple(
            AnthropicMessagesToolResultParam(
                type="tool_result",
                tool_use_id=str(result.get("tool_call_id") or ""),
                content=str(result.get("result") or ""),
            )
            for result in tool_results
        ),
    )


async def anthropic_messages_with_mcp(
    max_tokens: int,
    messages: Sequence[Mapping[str, Any]],
    model: str,
    tools: Union[Sequence[Mapping[str, Any]], None] = None,
    **kwargs: Any,  # kwargs-ok: forwarded verbatim to litellm.anthropic_messages, which owns the param contract
) -> Union[AnthropicMessagesResponse, AsyncIterator[Any]]:
    """
    Expand litellm_proxy MCP references for `/v1/messages` and run the tool loop.

    The MCP gateway owns the expansion so the reference resolves against the
    caller's own credentials and access control, rather than being handed to the
    upstream provider as a url it cannot reach.
    """
    import litellm
    from litellm.experimental_mcp_client.tools import (
        transform_mcp_tool_to_anthropic_tool,
    )
    from litellm.responses.mcp.litellm_proxy_mcp_handler import (
        LiteLLM_Proxy_MCP_Handler,
    )

    mcp_references, other_tools = LiteLLM_Proxy_MCP_Handler._parse_mcp_tools(tools)

    if not mcp_references:
        return await litellm.anthropic_messages(
            max_tokens=max_tokens,
            messages=list(messages),
            model=model,
            tools=list(tools) if tools else None,
            _skip_mcp_handler=True,
            **kwargs,
        )

    context = MCPRequestContext.resolve(kwargs=dict(kwargs), tools=tools)

    (
        deduplicated_mcp_tools,
        tool_server_map,
    ) = await LiteLLM_Proxy_MCP_Handler._process_mcp_tools_without_openai_transform(
        context.user_api_key_auth,
        mcp_references,
        litellm_trace_id=context.litellm_trace_id,
        mcp_auth_header=context.mcp_auth_header,
        mcp_server_auth_headers=context.mcp_server_auth_headers,
        request_tags=list(context.request_tags) if context.request_tags else None,
    )

    anthropic_tools: Sequence[AnthropicMessagesTool] = tuple(
        transform_mcp_tool_to_anthropic_tool(mcp_tool) for mcp_tool in deduplicated_mcp_tools
    )
    all_tools = [*anthropic_tools, *(other_tools or ())]

    should_auto_execute = LiteLLM_Proxy_MCP_Handler._should_auto_execute_tools(
        mcp_tools_with_litellm_proxy=mcp_references
    )
    stream = bool(kwargs.pop("stream", False))

    base_call_args: Mapping[str, Any] = {
        "max_tokens": max_tokens,
        "model": model,
        "tools": all_tools or None,
        "_skip_mcp_handler": True,
        **kwargs,
    }

    if not should_auto_execute:
        return await litellm.anthropic_messages(messages=list(messages), stream=stream, **base_call_args)

    working_messages: Sequence[Mapping[str, Any]] = tuple(messages)
    response: AnthropicMessagesResponse = await litellm.anthropic_messages(
        messages=list(working_messages), stream=False, **base_call_args
    )

    for _ in range(MAX_MCP_TOOL_USE_ITERATIONS):
        if _get_stop_reason(response) != "tool_use":
            break

        tool_use_blocks = _extract_tool_use_blocks(response)
        if not tool_use_blocks:
            break

        tool_results = await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
            tool_server_map=tool_server_map,
            tool_calls=list(tool_use_blocks),
            user_api_key_auth=context.user_api_key_auth,
            mcp_auth_header=context.mcp_auth_header,
            mcp_server_auth_headers=context.mcp_server_auth_headers,
            oauth2_headers=context.oauth2_headers,
            raw_headers=context.raw_headers,
            litellm_call_id=context.litellm_call_id,
            litellm_trace_id=context.litellm_trace_id,
            request_tags=list(context.request_tags) if context.request_tags else None,
        )

        # Every tool call was skipped, so there is nothing to feed back; a
        # tool_result message with empty content is rejected by Anthropic.
        if not tool_results:
            break

        working_messages = (
            *working_messages,
            {"role": "assistant", "content": list(_get_response_content(response))},
            _build_tool_result_message(tool_results),
        )
        response = await litellm.anthropic_messages(messages=list(working_messages), stream=False, **base_call_args)
    else:
        verbose_logger.warning(
            f"MCP tool loop hit its {MAX_MCP_TOOL_USE_ITERATIONS} iteration cap for model {model}; "
            "returning the last response"
        )

    if stream:
        from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
            FakeAnthropicMessagesStreamIterator,
        )

        return FakeAnthropicMessagesStreamIterator(response)
    return response
