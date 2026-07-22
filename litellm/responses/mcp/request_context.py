"""
The per-request context an MCP gateway handler needs.

Listing and executing MCP tools both need the caller's identity, their MCP auth
headers, and the request's trace/tag identifiers. Every gateway surface resolves
the same set from its own kwargs, so resolving it in one place keeps a new
surface from silently dropping a field: omitting the auth headers, for instance,
still executes the tool, just with no credentials.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Union


@dataclass(frozen=True, slots=True)
class MCPRequestContext:
    """Everything a gateway handler must forward to MCP tool listing and execution."""

    user_api_key_auth: Any  # any-ok: UserAPIKeyAuth is proxy-only; importing it here would create a cycle
    mcp_auth_header: Union[str, None] = None
    mcp_server_auth_headers: Union[Mapping[str, Mapping[str, str]], None] = None
    oauth2_headers: Union[Mapping[str, str], None] = None
    raw_headers: Union[Mapping[str, str], None] = None
    request_tags: Union[Sequence[str], None] = None
    litellm_trace_id: Union[str, None] = None
    litellm_call_id: Union[str, None] = None

    @classmethod
    def resolve(
        cls,
        kwargs: Mapping[str, Any],
        tools: Union[Iterable[Any], None],
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

        litellm_metadata = kwargs.get("litellm_metadata") or {}
        metadata = kwargs.get("metadata") or {}
        user_api_key_auth = (
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
            request_tags=LiteLLM_Proxy_MCP_Handler._get_parent_request_tags(dict(kwargs)),
            litellm_trace_id=kwargs.get("litellm_trace_id"),
            litellm_call_id=kwargs.get("litellm_call_id"),
        )
