"""Agentic tool loop for completion-bridge A2A agents.

Drives the model in a loop (model -> tool/agent calls -> execute -> feed results
back -> repeat) until it produces a final answer. An agent can call MCP tools
and delegate to other A2A agents; the inter-agent hop is a real A2A message/send
supplied by the proxy layer via ``call_agent``.
"""

import json
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional

import litellm
from litellm._logging import verbose_logger

AGENT_TOOL_PREFIX = "agent__"
DEFAULT_MAX_ITERATIONS = 8


def _mcp_results_to_tool_messages(
    tool_results: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """``_execute_tool_calls`` returns ``{tool_call_id, result, name}``, but the
    chat API needs ``{role: "tool", tool_call_id, content}``; without this
    conversion the model never sees the result and re-calls the tool until the
    loop is exhausted, yielding an empty final answer."""
    return tuple(
        {
            "role": "tool",
            "tool_call_id": result.get("tool_call_id"),
            "content": str(result.get("result", "")),
        }
        for result in tool_results
    )


def _build_agent_tools(callable_agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One function tool per callable agent. The model picks an agent by name;
    the tool name carries the target agent id so we can route the A2A hop."""
    tools: list[dict[str, Any]] = []
    for agent in callable_agents:
        agent_id = agent.get("agent_id") or agent.get("id")
        if not agent_id:
            continue
        name = agent.get("name") or agent_id
        description = (
            agent.get("description")
            or f"Delegate a self-contained task to the '{name}' specialist agent and use its reply."
        )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"{AGENT_TOOL_PREFIX}{agent_id}",
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The complete instruction and all facts the specialist needs; it has no prior context.",
                            }
                        },
                        "required": ["message"],
                    },
                },
            }
        )
    return tools


async def _load_mcp_tools(
    user_api_key_auth: Any,
    litellm_trace_id: Optional[str],
):
    """Returns (openai_tools, tool_server_map) for the agent's allowed MCP servers."""
    from litellm.responses.mcp.litellm_proxy_mcp_handler import (
        LiteLLM_Proxy_MCP_Handler,
    )

    mcp_tools_spec = [
        {"type": "mcp", "server_url": "litellm_proxy", "require_approval": "never"}
    ]
    (
        deduplicated,
        tool_server_map,
    ) = await LiteLLM_Proxy_MCP_Handler._process_mcp_tools_without_openai_transform(
        user_api_key_auth=user_api_key_auth,
        mcp_tools_with_litellm_proxy=mcp_tools_spec,
        litellm_trace_id=litellm_trace_id,
    )
    openai_tools = LiteLLM_Proxy_MCP_Handler._transform_mcp_tools_to_openai(
        deduplicated, target_format="chat"
    )
    return openai_tools, tool_server_map


async def run_agentic_loop(
    *,
    model: str,
    messages: list[dict[str, Any]],
    completion_kwargs: dict[str, Any],
    callable_agents: list[dict[str, Any]],
    call_agent: Callable[[str, str], Awaitable[str]],
    user_api_key_auth: Any = None,
    enable_mcp_tools: bool = False,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
):
    """
    Drive the model until it stops requesting tools/agents.

    - Agent (``agent__<id>``) tool calls -> real A2A hop via ``call_agent``.
    - MCP tool calls -> executed against the proxy MCP manager (auth-scoped).
    Returns the final ModelResponse.
    """
    agent_tools = _build_agent_tools(callable_agents)

    mcp_openai_tools: list[dict[str, Any]] = []
    tool_server_map: dict[str, str] = {}
    if enable_mcp_tools:
        mcp_openai_tools, tool_server_map = await _load_mcp_tools(
            user_api_key_auth=user_api_key_auth,
            litellm_trace_id=completion_kwargs.get("litellm_trace_id"),
        )

    all_tools = (mcp_openai_tools + agent_tools) or None
    messages = list(messages)
    response = None

    for _ in range(max_iterations):
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            tools=all_tools,
            stream=False,
            **completion_kwargs,
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            break

        messages.append(
            message.model_dump() if hasattr(message, "model_dump") else message
        )

        agent_calls = [
            tc
            for tc in tool_calls
            if (tc.function.name or "").startswith(AGENT_TOOL_PREFIX)
        ]
        mcp_calls = [tc for tc in tool_calls if tc not in agent_calls]

        if mcp_calls:
            from litellm.responses.mcp.litellm_proxy_mcp_handler import (
                LiteLLM_Proxy_MCP_Handler,
            )

            tool_results = await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
                tool_server_map=tool_server_map,
                tool_calls=mcp_calls,
                user_api_key_auth=user_api_key_auth,
            )
            messages.extend(_mcp_results_to_tool_messages(tool_results))

        for tool_call in agent_calls:
            target_agent_id = (tool_call.function.name or "")[len(AGENT_TOOL_PREFIX) :]
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            sub_message = arguments.get("message", "")
            try:
                reply = await call_agent(target_agent_id, sub_message)
            except Exception as e:
                verbose_logger.warning(
                    f"A2A sub-agent call to '{target_agent_id}' failed: {e}"
                )
                reply = f"Error calling agent '{target_agent_id}': {e}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": reply,
                }
            )

    # If the loop ended on a tool-calling turn (no prose), make one tool-free
    # call so the agent summarises instead of returning an empty message.
    final_message = response.choices[0].message if response is not None else None
    final_text = getattr(final_message, "content", None) if final_message else None
    if not (final_text or "").strip():
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            stream=False,
            **completion_kwargs,
        )

    return response
