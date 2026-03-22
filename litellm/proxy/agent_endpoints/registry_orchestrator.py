"""
RegistryOrchestrator — centralises all registry-backed orchestration logic.

Responsibilities
----------------
- Parsing ``a2a_agent`` tool configs out of a request's ``tools`` list.
- Resolving registered A2A agents from the global agent registry and wrapping each
  one as an OpenAI function tool that the LLM can call.
- Executing a single A2A tool call via JSON-RPC 2.0 ``message/send``.
- Applying the semantic MCP tool filter when the caller opts in via
  ``"semantic_filter": true`` on an MCP tool config.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from litellm._logging import verbose_logger

# NOTE: Kept broad to avoid coupling to optional OpenAI SDK typing symbols.
ToolParam = Any

LITELLM_PROXY_AGENTS_URL = "litellm_proxy/agents"

# Import hoisted out of the callback loop to avoid re-evaluating on every iteration.
try:
    from litellm.proxy.hooks.mcp_semantic_filter.hook import (  # noqa: E501
        SemanticToolFilterHook as _SemanticToolFilterHook,
    )
except ImportError:
    _SemanticToolFilterHook = None  # type: ignore


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _parse_a2a_response(data: Dict[str, Any]) -> str:
    """Extract text content from an A2A JSON-RPC message/send response.

    Handles both ``"type": "text"`` (older A2A SDK) and ``"kind": "text"``
    (A2A SDK >= 0.3) part schemas.
    """
    if "error" in data:
        err = data["error"]
        return f"Agent error: {err.get('message', str(err))}"

    result = data.get("result", {})

    def _is_text_part(p: Dict[str, Any]) -> bool:
        return (p.get("kind") == "text" or p.get("type") == "text") and bool(
            p.get("text")
        )

    # A2A spec: result.artifacts[].parts[].text
    for artifact in result.get("artifacts", []):
        texts = [p["text"] for p in artifact.get("parts", []) if _is_text_part(p)]
        if texts:
            return "\n".join(texts)

    # Fallback: status.message.parts[].text
    status = result.get("status", {})
    if isinstance(status, dict):
        msg = status.get("message") or {}
        for p in msg.get("parts", []):
            if _is_text_part(p):
                return p["text"]

    return str(result) if result else "Agent executed successfully"


# ---------------------------------------------------------------------------
# RegistryOrchestrator
# ---------------------------------------------------------------------------


class RegistryOrchestrator:
    """
    Static-method class that owns all registry-backed orchestration concerns:

    * Parsing A2A agent tool configs from a request.
    * Resolving registered agents and wrapping them as function tools.
    * Executing A2A tool calls via JSON-RPC.
    * Applying the per-request semantic MCP tool filter.
    """

    @staticmethod
    def parse_agent_tool_configs(
        tools: Optional[Iterable[ToolParam]],
    ) -> Tuple[List[ToolParam], List[Any]]:
        """
        Separate ``a2a_agent`` registry tool configs from all other tools.

        Returns:
            (agent_tool_configs, other_tools)
        """
        agent_tool_configs: List[ToolParam] = []
        other_tools: List[Any] = []

        if tools:
            for tool in tools:
                if isinstance(tool, dict) and tool.get("type") == "a2a_agent":
                    server_url = tool.get("server_url", "")
                    if (
                        isinstance(server_url, str)
                        and server_url == LITELLM_PROXY_AGENTS_URL
                    ):
                        agent_tool_configs.append(tool)
                    else:
                        other_tools.append(tool)
                else:
                    other_tools.append(tool)

        return agent_tool_configs, other_tools

    @staticmethod
    async def resolve_agent_tools(
        user_api_key_auth: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, str]]]:
        """
        Read all registered A2A agents and expose each as an OpenAI function tool.

        Returns:
            (function_tools, agent_tool_map)

            * ``function_tools``: list of ``{"type": "function", "function": {...}}`` dicts
            * ``agent_tool_map``: mapping of sanitized function name → ``{"url": str, "agent_name": str}``
        """
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        # NOTE: user_api_key_auth is accepted for future per-key agent filtering
        # (mirroring get_allowed_mcp_servers). Agent-level access control is not yet
        # implemented in AgentRegistry; all registered agents are returned for now.
        _ = user_api_key_auth

        agents = global_agent_registry.get_agent_list()
        function_tools: List[Dict[str, Any]] = []
        agent_tool_map: Dict[str, Dict[str, str]] = {}

        for agent in agents:
            card = agent.agent_card_params or {}
            agent_url = card.get("url", "")
            agent_name = card.get("name") or agent.agent_name

            if not agent_url:
                verbose_logger.warning(
                    "Agent '%s' has no URL configured, skipping", agent_name
                )
                continue

            description = card.get("description") or f"A2A agent: {agent_name}"

            # Enrich description with up to 3 skill descriptions
            skills = card.get("skills") or []
            skill_descs = [
                s.get("description", "")
                for s in skills[:3]
                if isinstance(s, dict) and s.get("description")
            ]
            if skill_descs:
                description += " Skills: " + "; ".join(skill_descs)

            # Sanitize to a valid OpenAI function name (^[a-zA-Z0-9_-]{1,64}$)
            func_name = (
                re.sub(r"[^a-zA-Z0-9_-]", "_", agent_name)[:64]
                or f"agent_{agent.agent_id[:8]}"
            )

            # Deduplicate: if two agents produce the same sanitized name, append the
            # agent_id suffix so neither is silently dropped.
            if func_name in agent_tool_map:
                func_name = f"{func_name}_{agent.agent_id[:8]}"[:64]
                verbose_logger.warning(
                    "Agent name collision: renamed to '%s' to avoid overwrite",
                    func_name,
                )

            function_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "The message or task to send to this agent",
                                }
                            },
                            "required": ["message"],
                            "additionalProperties": False,
                        },
                    },
                }
            )
            agent_tool_map[func_name] = {"url": agent_url, "agent_name": agent_name}

        verbose_logger.debug(
            "Wrapped %d registered agents as function tools: %s",
            len(function_tools),
            list(agent_tool_map.keys()),
        )
        return function_tools, agent_tool_map

    @staticmethod
    async def execute_a2a_tool_call(
        agent_url: str,
        agent_name: str,
        message: str,
        tool_call_id: str,
        tool_name: str,
        litellm_trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a message to an A2A agent via LiteLLM's asend_message and return the result."""
        import uuid

        try:
            from a2a.types import (
                Message,
                MessageSendParams,
                Part,
                Role,
                SendMessageRequest,
                TextPart,
            )
        except ImportError as exc:
            raise ImportError(
                "The 'a2a' package is required for A2A agent calls. "
                "Install it with: pip install a2a-sdk"
            ) from exc

        from litellm.a2a_protocol.main import asend_message

        a2a_message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=message))],
            message_id=uuid.uuid4().hex,
            context_id=litellm_trace_id,
        )
        request = SendMessageRequest(
            id=str(uuid.uuid4()),
            params=MessageSendParams(message=a2a_message),
        )

        try:
            response = await asend_message(
                api_base=agent_url,
                request=request,
                agent_id=agent_name,
            )
            result_text = _parse_a2a_response(
                response.model_dump(mode="json", exclude_none=True)
            )
            verbose_logger.debug(
                "A2A agent '%s' returned: %s", agent_name, result_text[:200]
            )
            return {
                "tool_call_id": tool_call_id,
                "result": result_text,
                "name": tool_name,
            }
        except Exception as e:
            verbose_logger.exception("Error calling A2A agent '%s': %s", agent_name, e)
            return {
                "tool_call_id": tool_call_id,
                "result": f"Error calling agent {agent_name}: {str(e)}",
                "name": tool_name,
            }

    @staticmethod
    async def apply_semantic_filter(
        tools: List[Any],
        messages: List[Any],
    ) -> List[Any]:
        """
        Filter MCP tools semantically based on the user query.

        Uses the global ``SemanticToolFilterHook`` if configured; otherwise returns
        all tools unchanged.
        """
        try:
            import litellm

            for callback in litellm.callbacks or []:
                if _SemanticToolFilterHook is None:
                    break
                if isinstance(callback, _SemanticToolFilterHook):
                    query = callback.filter.extract_user_query(messages)
                    if query:
                        filtered = await callback.filter.filter_tools(
                            query=query,
                            available_tools=tools,
                        )
                        verbose_logger.debug(
                            "Semantic filter (per-tool flag): %d → %d tools for query '%s...'",
                            len(tools),
                            len(filtered),
                            query[:60],
                        )
                        return filtered
        except Exception as e:
            verbose_logger.warning(
                "semantic_filter flag: filter failed (%s), using all tools", e
            )
        return tools
