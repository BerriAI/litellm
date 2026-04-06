"""
MCP End User Permission Guardrail Hook

Enforces end user permissions for MCP server access via apply_guardrail:
- input_type="request" → filter tools the end user cannot access

Permission logic:
- No end_user_id               → allow all (key/team-level permissions apply)
- end_user_id, no mcp_servers  → allow all (default)
- end_user_id + mcp_servers    → allow only those servers
"""

from typing import TYPE_CHECKING, Any, List, Literal, Optional, Type

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import LiteLLM_ObjectPermissionTable
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

GUARDRAIL_NAME = "mcp_end_user_permission"


class MCPEndUserPermissionGuardrail(CustomGuardrail):
    """
    Guardrail that enforces end user permissions for MCP server access.

    Runs on input only (pre-call). Filters tools in the request that the
    end user is not permitted to call based on their object_permission.

    end_user_object_permission is populated on UserAPIKeyAuth during auth.
    The guardrail resolves it via a cached get_end_user_object lookup —
    no extra DB round-trip when the cache is warm.
    """

    def __init__(self, **kwargs):
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
            ]
        super().__init__(**kwargs)
        verbose_proxy_logger.debug("MCP End User Permission Guardrail initialized")

    # ------------------------------------------------------------------
    # apply_guardrail — filters MCP tools on the request side only
    # ------------------------------------------------------------------

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"] = "request",
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Filters MCP tools the end user cannot access based on their
        object_permission.mcp_servers / mcp_access_groups settings.
        """
        object_permission = await self._resolve_end_user_object_permission(request_data)
        return await self._check_request_tools(inputs, object_permission)

    # ------------------------------------------------------------------
    # Private — request-side tool filtering
    # ------------------------------------------------------------------

    async def _check_request_tools(
        self,
        inputs: GenericGuardrailAPIInputs,
        object_permission: Optional[LiteLLM_ObjectPermissionTable],
    ) -> GenericGuardrailAPIInputs:
        tools = inputs.get("tools")
        if not tools:
            return inputs

        allowed_mcp_servers = (
            await self._get_allowed_mcp_servers_from_object_permission(
                object_permission
            )
        )
        if allowed_mcp_servers is None:
            return inputs  # No restrictions → pass through unchanged

        verbose_proxy_logger.debug(
            f"MCP guardrail: end user restricted to MCP servers: {allowed_mcp_servers}"
        )

        filtered_tools = []
        removed_tools = []

        for tool in tools:
            tool_name = self._get_tool_name_from_definition(tool)
            server_name = (
                self._extract_mcp_server_name(tool_name) if tool_name else None
            )

            if server_name is None:
                # Not an MCP tool (no prefix) or unrecognised format → keep
                filtered_tools.append(tool)
            elif server_name in allowed_mcp_servers:
                filtered_tools.append(tool)
            else:
                removed_tools.append(tool_name)
                verbose_proxy_logger.warning(
                    f"MCP guardrail: removing tool '{tool_name}' "
                    f"(server: '{server_name}') — not in end user's allowed servers"
                )

        if removed_tools:
            verbose_proxy_logger.debug(
                f"MCP guardrail: removed {len(removed_tools)} unauthorized MCP tool(s): {removed_tools}"
            )
            inputs["tools"] = filtered_tools

        return inputs

    # ------------------------------------------------------------------
    # Private — end user permission resolution
    # ------------------------------------------------------------------

    @staticmethod
    async def _resolve_end_user_object_permission(
        request_data: dict,
    ) -> Optional[LiteLLM_ObjectPermissionTable]:
        """
        Resolve the end user's object_permission via the cached auth lookup.

        Uses get_end_user_object (same path as auth) so no extra DB round-trip
        when the cache is warm.
        """
        end_user_id = MCPEndUserPermissionGuardrail._get_end_user_id_from_request_data(
            request_data
        )
        if not end_user_id:
            return None

        end_user_object = await MCPEndUserPermissionGuardrail._fetch_end_user_object(
            end_user_id
        )
        return (
            end_user_object.object_permission if end_user_object is not None else None
        )

    @staticmethod
    def _get_end_user_id_from_request_data(request_data: dict) -> Optional[str]:
        return request_data.get("user_api_key_end_user_id") or request_data.get(
            "litellm_metadata", {}
        ).get("user_api_key_end_user_id")

    @staticmethod
    async def _fetch_end_user_object(end_user_id: str):  # type: ignore[return]
        """
        Fetch end user object via the same cached path used during auth.
        No extra DB round-trip when the cache is warm.
        """
        from litellm.proxy.auth.auth_checks import get_end_user_object
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if prisma_client is None:
            return None

        try:
            return await get_end_user_object(
                end_user_id=end_user_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=None,
                proxy_logging_obj=proxy_logging_obj,
                route="/mcp",
            )
        except Exception as e:
            verbose_proxy_logger.warning(
                f"MCP guardrail: failed to fetch end_user_object for '{end_user_id}': {e}"
            )
            return None

    # ------------------------------------------------------------------
    # Private — permission derivation
    # ------------------------------------------------------------------

    @staticmethod
    async def _get_allowed_mcp_servers_from_object_permission(
        object_permission: Optional[LiteLLM_ObjectPermissionTable],
    ) -> Optional[List[str]]:
        """
        Returns:
            None  — no restrictions configured, allow all MCP servers
            list  — restrict to exactly these server names
        """
        if object_permission is None:
            return None

        direct_mcp_servers = object_permission.mcp_servers or []
        mcp_access_groups = object_permission.mcp_access_groups or []

        if not direct_mcp_servers and not mcp_access_groups:
            return None  # Both empty → no restrictions

        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )

        access_group_servers = (
            await MCPRequestHandler._get_mcp_servers_from_access_groups(
                mcp_access_groups
            )
        )

        return list(set(direct_mcp_servers + access_group_servers))

    # ------------------------------------------------------------------
    # Config model — exposes this guardrail in the UI
    # ------------------------------------------------------------------

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.mcp_end_user_permission import (
            MCPEndUserPermissionGuardrailConfigModel,
        )

        return MCPEndUserPermissionGuardrailConfigModel

    # ------------------------------------------------------------------
    # Private — tool name extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_mcp_server_name(tool_name: str) -> Optional[str]:
        """
        Split "github-create_issue" → "github".
        Returns None if the tool name has no '-' prefix (not an MCP tool).
        """
        if not tool_name or "-" not in tool_name:
            return None
        return tool_name.split("-", 1)[0]

    @staticmethod
    def _get_tool_name_from_definition(tool: Any) -> Optional[str]:
        """
        Extract tool name from a definition dict.

        OpenAI format:   {"type": "function", "function": {"name": "..."}}
        Anthropic format: {"name": "...", "input_schema": {...}}
        """
        if not isinstance(tool, dict):
            return None
        function_def = tool.get("function")
        if isinstance(function_def, dict):
            name = function_def.get("name")
            if name:
                return name
        return tool.get("name")
