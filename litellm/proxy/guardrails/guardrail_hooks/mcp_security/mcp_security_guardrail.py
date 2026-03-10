"""
MCP Security Guardrail for LiteLLM.

Validates that MCP servers referenced in request tools are registered
on the LiteLLM gateway. Blocks or alerts when unregistered servers are found.
"""

from typing import Any, List, Literal, Optional, Set, Union

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.responses.mcp.litellm_proxy_mcp_handler import (
    LITELLM_PROXY_MCP_SERVER_URL_PREFIX,
)
from litellm.types.guardrails import GuardrailEventHooks


class MCPSecurityGuardrail(CustomGuardrail):
    def __init__(
        self,
        on_violation: Literal["block", "alert"] = "block",
        **kwargs,
    ):
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [GuardrailEventHooks.pre_call]
        super().__init__(**kwargs)
        self.on_violation = on_violation

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.pre_call
            )
            is not True
        ):
            return data

        unregistered = self._find_unregistered_mcp_servers(data)
        if not unregistered:
            return data

        message = (
            f"MCP Security: request references unregistered MCP server(s): "
            f"{', '.join(sorted(unregistered))}. "
            f"Only servers registered on this gateway are allowed."
        )

        if self.on_violation == "block":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated guardrail policy",
                    "guardrail": "mcp_security",
                    "unregistered_servers": sorted(unregistered),
                    "detection_message": message,
                },
            )
        else:
            verbose_proxy_logger.warning(message)

        return data

    @staticmethod
    def _extract_mcp_server_names_from_tools(tools: List[dict]) -> Set[str]:
        """Extract MCP server names from tools with type=mcp and litellm_proxy server_url."""
        server_names: Set[str] = set()
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") != "mcp":
                continue
            server_url = tool.get("server_url", "")
            if not isinstance(server_url, str):
                continue
            if server_url.startswith(LITELLM_PROXY_MCP_SERVER_URL_PREFIX):
                name = server_url[len(LITELLM_PROXY_MCP_SERVER_URL_PREFIX):]
                if name:
                    server_names.add(name)
        return server_names

    @staticmethod
    def _find_unregistered_mcp_servers(data: dict) -> Set[str]:
        """Check tools in data against the MCP server registry. Returns set of unregistered server names."""
        tools = data.get("tools")
        if not tools or not isinstance(tools, list):
            return set()

        requested_servers = (
            MCPSecurityGuardrail._extract_mcp_server_names_from_tools(tools)
        )
        if not requested_servers:
            return set()

        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        registry = global_mcp_server_manager.get_registry()
        registered_names = set(registry.keys())

        return requested_servers - registered_names
