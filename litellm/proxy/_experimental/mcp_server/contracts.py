from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from types import EllipsisType, MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias

from mcp.types import ErrorData, InputRequest, InputResponse, InputResponses
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server.tool_outcome import WireCompat
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer

if TYPE_CHECKING:
    from mcp.types import (
        ListPromptsRequest,
        ListPromptsResult,
        ListResourcesRequest,
        ListResourcesResult,
        ListResourceTemplatesRequest,
        ListResourceTemplatesResult,
        ListToolsRequest,
        ListToolsResult,
    )

    CatalogListRequest: TypeAlias = (
        ListToolsRequest | ListPromptsRequest | ListResourcesRequest | ListResourceTemplatesRequest
    )
    CatalogListResult: TypeAlias = (
        ListToolsResult | ListPromptsResult | ListResourcesResult | ListResourceTemplatesResult
    )

    from litellm.proxy._experimental.mcp_server.server_resolution import ResolvedMCPServer


class TargetCatalog(Protocol):
    async def list(self, context: OperationContext, request: CatalogListRequest) -> CatalogListResult: ...

    async def resolve(
        self,
        server_id: str,
        caller: UserAPIKeyAuth,
        *,
        is_admin_view: bool,
        not_found_detail: Mapping[str, str],
        forbidden_detail: Mapping[str, str],
        non_admin_missing: Literal["not_found", "forbidden"],
    ) -> ResolvedMCPServer: ...


def copy_caller(auth: UserAPIKeyAuth | None) -> UserAPIKeyAuth | None:
    if auth is None:
        return None
    span: Final = auth.parent_otel_span
    return deepcopy(auth, {id(span): span} if span is not None else None)


@dataclass(frozen=True, slots=True)
class OperationContext:
    _caller: UserAPIKeyAuth | None = field(repr=False)
    mcp_auth_header: str | None = field(default=None, repr=False)
    mcp_servers: tuple[str, ...] | None = None
    mcp_server_auth_headers: Mapping[str, Mapping[str, str]] | None = field(default=None, repr=False)
    oauth2_headers: Mapping[str, str] | None = field(default=None, repr=False)
    raw_headers: Mapping[str, str] | None = field(default=None, repr=False)
    client_ip: str | None = None
    mcp_proxy_mode: bool = False
    wire_compat: WireCompat = WireCompat.LEGACY
    protocol_version: str | None = None
    incoming_bearer_token: SecretStr | None | EllipsisType = field(default=..., repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_caller", copy_caller(self._caller))
        object.__setattr__(self, "mcp_servers", tuple(self.mcp_servers) if self.mcp_servers is not None else None)
        object.__setattr__(
            self,
            "oauth2_headers",
            MappingProxyType(dict(self.oauth2_headers)) if self.oauth2_headers is not None else None,
        )
        object.__setattr__(
            self, "raw_headers", MappingProxyType(dict(self.raw_headers)) if self.raw_headers is not None else None
        )
        object.__setattr__(
            self,
            "mcp_server_auth_headers",
            MappingProxyType(
                {key: MappingProxyType(dict(value)) for key, value in self.mcp_server_auth_headers.items()}
            )
            if self.mcp_server_auth_headers is not None
            else None,
        )

    @property
    def user_api_key_auth(self) -> UserAPIKeyAuth | None:
        return copy_caller(self._caller)

    def legacy_auth(
        self,
    ) -> tuple[
        UserAPIKeyAuth | None,
        str | None,
        list[str] | None,
        dict[str, dict[str, str]] | None,
        dict[str, str] | None,
        dict[str, str] | None,
        str | None,
    ]:
        return (
            self.user_api_key_auth,
            self.mcp_auth_header,
            list(self.mcp_servers) if self.mcp_servers is not None else None,
            {key: dict(value) for key, value in self.mcp_server_auth_headers.items()}
            if self.mcp_server_auth_headers is not None
            else None,
            dict(self.oauth2_headers) if self.oauth2_headers is not None else None,
            dict(self.raw_headers) if self.raw_headers is not None else None,
            self.client_ip,
        )


class ProgressCallback(Protocol):
    async def __call__(self, progress: float, total: float | None, /) -> None: ...


class ClientInteraction(Protocol):
    async def request(self, key: str, request: InputRequest) -> InputResponse | ErrorData: ...


@dataclass(frozen=True, slots=True)
class AuthorizedToolCall:
    name: str
    arguments: Mapping[str, object]
    allowed_mcp_servers: tuple[MCPServer, ...]
    start_time: datetime
    host_progress_callback: ProgressCallback | None
    guardrail_context: Mapping[str, object] | None
    logging_data: Mapping[str, object]
    input_responses: InputResponses | None = None
    request_state: str | None = None
