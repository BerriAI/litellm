"""
The per-request context an MCP gateway handler needs.

Listing and executing MCP tools both need the caller's identity, their MCP auth
headers, and the request's trace/tag identifiers. Every gateway surface resolves
the same set from its own kwargs, so resolving it in one place keeps a new
surface from silently dropping a field: omitting the auth headers, for instance,
still executes the tool, just with no credentials.
"""

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from pydantic import TypeAdapter
from typing_extensions import NotRequired, ReadOnly, TypedDict

if TYPE_CHECKING:
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth


class _AuthCarryingMetadata(TypedDict):
    """The one key this module reads out of a request's ``metadata`` / ``litellm_metadata``."""

    user_api_key_auth: ReadOnly[NotRequired["UserAPIKeyAuth | None"]]


PROVIDER_MCP_ENFORCEMENT_SETTING: Final = "enforce_provider_mcp_permissions"


def provider_mcp_enforcement_enabled() -> bool:
    """Provider-native MCP checks are opt-in. Off (the default) forwards provider-run MCP attachments
    unchecked, exactly as before registrations existed. Read per request, so a config reload applies."""
    from litellm.proxy import proxy_server

    value: Final = proxy_server.general_settings.get(PROVIDER_MCP_ENFORCEMENT_SETTING)
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


async def authorize_provider_mcp_request(
    body: Mapping[str, object], context: Mapping[str, object], *, provider_bound: bool = False
) -> None:
    buckets: Final = tuple(
        TypeAdapter(dict[str, object]).validate_python(context[key])
        for key in ("litellm_metadata", "metadata")
        if isinstance(context.get(key), Mapping)
    )
    auth: Final = next((bucket["user_api_key_auth"] for bucket in buckets if "user_api_key_auth" in bucket), None)
    if auth is None or not provider_mcp_enforcement_enabled():
        return
    extra: Final = body.get("extra_body")
    effective: Final = (
        {**body, **TypeAdapter(dict[str, object]).validate_python(extra)} if isinstance(extra, Mapping) else body
    )
    tools: Final = (*_tool_list(effective.get("tools")), *_input_item_tools(effective.get("input")))
    has_native_tools: Final = any(
        isinstance(tool, Mapping)
        and TypeAdapter(dict[str, object]).validate_python(tool).get("type") in ("mcp", "mcp_toolset", "url")
        for tool in tools
    )
    if not has_native_tools and not effective.get("mcp_servers"):
        return
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

    if not isinstance(auth, UserAPIKeyAuth):
        return
    from litellm.proxy.mcp_server.auth.provider_mcp import authorize_provider_mcp

    await authorize_provider_mcp({**effective, "tools": list(tools)}, auth, provider_bound=provider_bound)


def _tool_list(value: object) -> tuple[object, ...]:
    return TypeAdapter(tuple[object, ...]).validate_python(value) if isinstance(value, (list, tuple)) else ()


def _input_item_tools(value: object) -> tuple[object, ...]:
    """Tool definitions carried inside Responses input items, such as ``tool_search_output``; the provider
    loads them like top-level tools, so they need the same admission."""
    return tuple(
        chain.from_iterable(
            _tool_list(TypeAdapter(dict[str, object]).validate_python(item).get("tools"))
            for item in _tool_list(value)
            if isinstance(item, Mapping)
        )
    )


@dataclass(frozen=True, slots=True)
class MCPRequestContext:
    """Everything a gateway handler must forward to MCP tool listing and execution."""

    user_api_key_auth: "UserAPIKeyAuth | None"
    mcp_auth_header: str | None = None
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None
    oauth2_headers: dict[str, str] | None = None
    raw_headers: dict[str, str] | None = None
    request_tags: list[str] | None = None
    litellm_trace_id: str | None = None
    litellm_call_id: str | None = None
    guardrail_context: Mapping[str, object] | None = None

    @classmethod
    def resolve(
        cls,
        kwargs: Mapping[str, Any],
        tools: Iterable[object] | None,
    ) -> "MCPRequestContext":
        """
        Build the context from a gateway handler's kwargs.

        ``user_api_key_auth`` is read from both metadata keys because routes differ:
        LITELLM_METADATA_ROUTES (``/v1/messages``, ``/responses``) carry it in
        ``litellm_metadata`` while ``/chat/completions`` uses ``metadata``.
        """
        from litellm.responses.mcp.litellm_proxy_mcp_handler import (
            LiteLLM_Proxy_MCP_Handler,
        )
        from litellm.responses.utils import ResponsesAPIRequestUtils

        litellm_metadata: Final[_AuthCarryingMetadata] = kwargs.get("litellm_metadata") or {}
        metadata: Final[_AuthCarryingMetadata] = kwargs.get("metadata") or {}
        user_api_key_auth: Final[UserAPIKeyAuth | None] = (
            kwargs.get("user_api_key_auth")
            or litellm_metadata.get("user_api_key_auth")
            or metadata.get("user_api_key_auth")
        )

        (
            mcp_auth_header,
            mcp_server_auth_headers,
            oauth2_headers,
            raw_headers,
        ) = ResponsesAPIRequestUtils.extract_mcp_headers_from_request(
            secret_fields=kwargs.get("secret_fields"),
            tools=tools,
        )

        return cls(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            request_tags=LiteLLM_Proxy_MCP_Handler.get_parent_request_tags(dict(kwargs)),
            litellm_trace_id=kwargs.get("litellm_trace_id"),
            litellm_call_id=kwargs.get("litellm_call_id"),
            guardrail_context=cls.resolve_guardrail_context(kwargs),
        )

    @staticmethod
    def resolve_guardrail_context(kwargs: Mapping[str, object]) -> Mapping[str, object]:
        metadata_keys: Final = (
            "guardrails",
            "guardrail_config",
            "_guardrail_pipelines",
            "_pipeline_managed_guardrails",
            "applied_policies",
            "policy_sources",
            "tags",
        )
        buckets: Final = tuple(
            TypeAdapter(dict[str, object]).validate_python(kwargs[key])
            for key in ("litellm_metadata", "metadata")
            if isinstance(kwargs.get(key), Mapping)
        )
        sources: Final = (*buckets, kwargs)
        metadata: Final = MappingProxyType(
            {
                **MappingProxyType(
                    {
                        key: deepcopy(value)
                        for bucket in buckets
                        for key, value in bucket.items()
                        if key in metadata_keys
                    }
                ),
                "guardrails": deepcopy(
                    tuple(
                        selection
                        for source in sources
                        for selection in TypeAdapter(list[object]).validate_python(source.get("guardrails") or ())
                    )
                ),
                "guardrail_config": deepcopy(
                    {
                        key: value
                        for source in sources
                        for key, value in TypeAdapter(dict[str, object])
                        .validate_python(source.get("guardrail_config") or MappingProxyType({}))
                        .items()
                    }
                ),
            }
        )
        return MappingProxyType(
            {
                **MappingProxyType({key: kwargs[key] for key in ("model",) if key in kwargs}),
                "metadata": metadata,
            }
        )
