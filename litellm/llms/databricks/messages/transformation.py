from collections.abc import Mapping
from typing import Final

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping, iter_items
from litellm.llms.anthropic.pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.databricks.common_utils import (
    DATABRICKS_AI_GATEWAY_MESSAGES_PATH,
    DATABRICKS_ANTHROPIC_MESSAGES_PATH,
    DatabricksBase,
)
from litellm.types.router import GenericLiteLLMParams

DATABRICKS_UNSUPPORTED_SERVER_TOOL_PREFIXES: Final = ("web_search_", "web_fetch_", "code_execution_")


def _is_unsupported_server_tool(tool: object) -> bool:
    tool_type: Final = (as_str_mapping(tool) or {}).get("type")
    return isinstance(tool_type, str) and tool_type.startswith(DATABRICKS_UNSUPPORTED_SERVER_TOOL_PREFIXES)


def _tool_name(tool: object) -> object:
    return (as_str_mapping(tool) or {}).get("name")


def _without_unsupported_server_tools(params: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(params.get("tools"), list):
        return dict(params)
    sent_tools: Final = tuple(iter_items(params.get("tools")))
    kept_tools: Final = [tool for tool in sent_tools if not _is_unsupported_server_tool(tool)]
    dropped_names: Final = frozenset(_tool_name(tool) for tool in sent_tools if _is_unsupported_server_tool(tool))
    forced_tool: Final = _tool_name(params.get("tool_choice"))
    keeps_tool_choice: Final = "tool_choice" in params and bool(kept_tools) and forced_tool not in dropped_names
    remaining: Final = {key: value for key, value in params.items() if key not in ("tools", "tool_choice")}
    return (
        remaining
        | ({"tools": kept_tools} if kept_tools else {})
        | ({"tool_choice": params["tool_choice"]} if keeps_tool_choice else {})
    )


class DatabricksAnthropicMessagesConfig(DatabricksBase, AnthropicMessagesConfig):
    """
    Forwards /v1/messages for Claude models to the workspace's native Anthropic
    route (``/serving-endpoints/anthropic/v1/messages`` for a serving endpoint,
    ``/ai-gateway/anthropic/v1/messages`` for a Unity Catalog model service), so
    Anthropic-only request fields and response fields (``safeguards`` and
    ``safeguard_results``) survive instead of being bridged through chat completions.
    """

    @property
    def custom_llm_provider(self) -> str | None:
        return "databricks"

    def handles_web_search_natively(self) -> bool:
        return False

    def validate_anthropic_messages_environment(
        self,
        headers: dict[str, str],  # mutable-ok: mirrors the base signature
        model: str,
        messages: list[object],  # mutable-ok: mirrors the base signature
        optional_params: dict[str, object],  # mutable-ok: mirrors the base signature
        litellm_params: dict[str, object],  # mutable-ok: mirrors the base signature
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict[str, str], str | None]:  # mutable-ok: mirrors the base signature
        messages_api_base, auth_headers = self.databricks_validate_environment(  # pyright: ignore[reportUnknownMemberType]  # the shared helper takes a bare dict
            api_key=self.resolve_api_key(api_key),
            api_base=self.resolve_api_base(api_base),
            endpoint_type="anthropic_messages",
            custom_endpoint=False,
            headers=headers,
            custom_user_agent=self.configured_user_agent(litellm_params),
        )
        workspace_api_base: Final = messages_api_base.removesuffix(DATABRICKS_ANTHROPIC_MESSAGES_PATH)
        normalized_headers: Final = {
            ("content-type" if key.lower() == "content-type" else key): value for key, value in auth_headers.items()
        }
        finalized_headers: Final[dict[str, str]] = self._finalize_messages_headers(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the shared helper returns a bare dict  # mutable-ok: becomes the outbound headers
            headers=normalized_headers,
            optional_params=optional_params,
            messages=messages,
        )
        return finalized_headers, workspace_api_base

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],  # mutable-ok: mirrors the base signature
        litellm_params: dict[str, object],  # mutable-ok: mirrors the base signature
        stream: bool | None = None,
    ) -> str:
        resolved_api_base: Final = self._get_api_base(api_base).rstrip("/")
        if self.uses_ai_gateway(model):
            return f"{self.workspace_root(resolved_api_base)}{DATABRICKS_AI_GATEWAY_MESSAGES_PATH}"
        if resolved_api_base.endswith(DATABRICKS_ANTHROPIC_MESSAGES_PATH):
            return resolved_api_base
        return f"{resolved_api_base}{DATABRICKS_ANTHROPIC_MESSAGES_PATH}"

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict[str, object]],  # mutable-ok: mirrors the base signature
        anthropic_messages_optional_request_params: dict[str, object],  # mutable-ok: mirrors the base signature
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: mirrors the base signature
    ) -> dict[str, object]:  # mutable-ok: mirrors the base signature
        drops_unsupported_params: Final = litellm.drop_params is True or litellm_params.drop_params is True
        request_params: Final = (
            _without_unsupported_server_tools(anthropic_messages_optional_request_params)
            if drops_unsupported_params
            else anthropic_messages_optional_request_params
        )
        anthropic_messages_request: Final[dict[str, object]] = super().transform_anthropic_messages_request(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the shared base returns a bare dict  # mutable-ok: becomes the outbound request body
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        self._normalize_system_role_messages(anthropic_messages_request, model=model)  # pyright: ignore[reportUnknownMemberType]  # the shared helper takes a bare dict
        self._remove_scope_from_cache_control(anthropic_messages_request)  # pyright: ignore[reportUnknownMemberType]  # the shared helper takes a bare dict
        return anthropic_messages_request
