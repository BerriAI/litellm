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
    def _find_unregistered_mcp_servers(data: dict) -> Set[str]:
        """Check MCP tools in data against the MCP server registry.

        A tool is considered unregistered if its server_label/name
        does not match any registered MCP server name or alias.
        """
        tools = data.get("tools")
        if not tools or not isinstance(tools, list):
            return set()

        mcp_tools = [
            t
            for t in tools
            if isinstance(t, dict) and t.get("type") == "mcp"
        ]
        if not mcp_tools:
            return set()

        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        registry = global_mcp_server_manager.get_registry()

        registered_identifiers: Set[str] = set()
        for key, srv in registry.items():
            registered_identifiers.add(key)
            if srv.name:
                registered_identifiers.add(srv.name)
            if srv.alias:
                registered_identifiers.add(srv.alias)
            if srv.server_name:
                registered_identifiers.add(srv.server_name)
            if srv.url:
                registered_identifiers.add(srv.url)

        verbose_proxy_logger.info(
            "MCP Security: registered_identifiers=%s", registered_identifiers
        )

        unregistered: Set[str] = set()
        for tool in mcp_tools:
            server_label = tool.get("server_label") or tool.get("server_url") or ""
            server_url = tool.get("server_url") or ""

            if not server_label and not server_url:
                continue

            label_registered = server_label in registered_identifiers
            url_registered = server_url in registered_identifiers

            if not label_registered and not url_registered:
                display = server_label or server_url
                unregistered.add(display)

        verbose_proxy_logger.info(
            "MCP Security: unregistered=%s", unregistered
        )
        return unregistered
