"""Helpers for handling MCP-aware `/chat/completions` requests."""

from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Optional,
    Union,
    cast,
)

from litellm.responses.mcp.litellm_proxy_mcp_handler import (
    LiteLLM_Proxy_MCP_Handler,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ToolParam
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

CompletionCallable = Callable[..., Awaitable[Union[ModelResponse, CustomStreamWrapper]]]

_CHAT_COMPLETION_CALL_ARG_KEYS = [
    "model",
    "messages",
    "functions",
    "function_call",
    "timeout",
    "temperature",
    "top_p",
    "n",
    "stream",
    "stream_options",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "modalities",
    "prediction",
    "audio",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "response_format",
    "seed",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "logprobs",
    "top_logprobs",
    "deployment_id",
    "reasoning_effort",
    "verbosity",
    "safety_identifier",
    "service_tier",
    "base_url",
    "api_version",
    "api_key",
    "model_list",
    "extra_headers",
    "thinking",
    "web_search_options",
    "shared_session",
]


def _build_call_args_from_context(call_context: Dict[str, Any]) -> Dict[str, Any]:
    """Build kwargs for `acompletion` from the `completion` call context."""

    call_args = {
        key: call_context.get(key)
        for key in _CHAT_COMPLETION_CALL_ARG_KEYS
        if key in call_context
    }
    additional_kwargs = dict(call_context.get("kwargs") or {})
    call_args.update(additional_kwargs)
    return call_args


async def _call_acompletion_internal(
    completion_callable: CompletionCallable, **call_args: Any
) -> Union[ModelResponse, CustomStreamWrapper]:
    """Invoke `acompletion` while skipping MCP interception to avoid recursion."""

    safe_args = dict(call_args)
    safe_args["_skip_mcp_handler"] = True
    safe_args.pop("acompletion", None)
    return await completion_callable(**safe_args)


async def handle_chat_completion_with_mcp(
    call_context: Dict[str, Any],
    completion_callable: CompletionCallable,
) -> Optional[Union[ModelResponse, CustomStreamWrapper]]:
    """Handle MCP-enabled tool execution for chat completion requests."""

    call_args = _build_call_args_from_context(call_context)

    tools = call_args.get("tools")
    if not tools:
        return None

    tools_for_mcp = cast(Optional[Iterable[ToolParam]], tools)

    if not LiteLLM_Proxy_MCP_Handler._should_use_litellm_mcp_gateway(
        tools=tools_for_mcp
    ):
        return None

    mcp_tools, _ = LiteLLM_Proxy_MCP_Handler._parse_mcp_tools(tools)
    if not mcp_tools:
        return None

    base_call_args = dict(call_args)

    user_api_key_auth = call_args.get("user_api_key_auth") or (
        (call_args.get("metadata", {}) or {}).get("user_api_key_auth")
    )
    (
        deduplicated_mcp_tools,
        tool_server_map,
    ) = await LiteLLM_Proxy_MCP_Handler._process_mcp_tools_without_openai_transform(
        user_api_key_auth=user_api_key_auth,
        mcp_tools_with_litellm_proxy=mcp_tools,
    )

    openai_tools = LiteLLM_Proxy_MCP_Handler._transform_mcp_tools_to_openai(
        deduplicated_mcp_tools,
        target_format="chat",
    )

    base_call_args["tools"] = openai_tools or None

    should_auto_execute = LiteLLM_Proxy_MCP_Handler._should_auto_execute_tools(
        mcp_tools_with_litellm_proxy=mcp_tools
    )

    (
        mcp_auth_header,
        mcp_server_auth_headers,
        oauth2_headers,
        raw_headers,
    ) = ResponsesAPIRequestUtils.extract_mcp_headers_from_request(
        secret_fields=base_call_args.get("secret_fields"),
        tools=tools,
    )

    if not should_auto_execute:
        return await _call_acompletion_internal(completion_callable, **base_call_args)

    mock_tool_calls = base_call_args.pop("mock_tool_calls", None)

    initial_call_args = dict(base_call_args)
    initial_call_args["stream"] = False
    if mock_tool_calls is not None:
        initial_call_args["mock_tool_calls"] = mock_tool_calls

    initial_response = await _call_acompletion_internal(
        completion_callable, **initial_call_args
    )
    if not isinstance(initial_response, ModelResponse):
        return initial_response

    tool_calls = LiteLLM_Proxy_MCP_Handler._extract_tool_calls_from_chat_response(
        response=initial_response
    )

    if not tool_calls:
        if base_call_args.get("stream"):
            retry_args = dict(base_call_args)
            retry_args["stream"] = call_args.get("stream")
            return await _call_acompletion_internal(completion_callable, **retry_args)
        return initial_response

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

    follow_up_messages = LiteLLM_Proxy_MCP_Handler._create_follow_up_messages_for_chat(
        original_messages=call_args.get("messages", []),
        response=initial_response,
        tool_results=tool_results,
    )

    follow_up_call_args = dict(base_call_args)
    follow_up_call_args["messages"] = follow_up_messages
    follow_up_call_args["stream"] = call_args.get("stream")

    return await _call_acompletion_internal(completion_callable, **follow_up_call_args)
