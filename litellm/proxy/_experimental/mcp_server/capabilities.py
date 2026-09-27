from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Final, Literal

from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.shared.exceptions import MCPError
from mcp.types import DiscoverResult, InitializeRequestParams, InitializeResult, ServerCapabilities
from mcp_types.methods import CLIENT_REQUESTS
from mcp_types.version import HANDSHAKE_PROTOCOL_VERSIONS, LATEST_HANDSHAKE_VERSION
from pydantic import TypeAdapter

from litellm.types.mcp import MCP_LEGACY_VERSIONS, MCPAdvertisedVersions, MCPLegacyVersion, MCPSpecVersion, MCPTransport

GATEWAY_OPERATIONS: Final = frozenset(
    {
        "tools/list",
        "tools/call",
        "prompts/list",
        "prompts/get",
        "resources/list",
        "resources/read",
        "resources/templates/list",
    }
)


@dataclass(frozen=True, slots=True)
class RevisionSupport:
    transports: frozenset[MCPTransport]
    operations: frozenset[str]
    results: frozenset[Literal["complete", "input_required"]]
    extensions: frozenset[str]
    completed: bool


REVISION_SUPPORT: Final[Mapping[str, RevisionSupport]] = MappingProxyType(
    {
        version.value: RevisionSupport(
            transports=frozenset(MCPTransport)
            if version.value in HANDSHAKE_PROTOCOL_VERSIONS
            else frozenset({MCPTransport.http, MCPTransport.stdio}),
            operations=frozenset(method for method in GATEWAY_OPERATIONS if (method, version.value) in CLIENT_REQUESTS),
            results=frozenset({"complete"})
            if version.value in HANDSHAKE_PROTOCOL_VERSIONS
            else frozenset({"complete", "input_required"}),
            extensions=frozenset(),
            completed=version.value in HANDSHAKE_PROTOCOL_VERSIONS,
        )
        for version in MCPSpecVersion
    }
)
_COMPLETED_REVISIONS: Final = tuple(version for version, support in REVISION_SUPPORT.items() if support.completed)
TRANSLATION_PAIRS: Final = frozenset(product(_COMPLETED_REVISIONS, repeat=2))
_ADVERTISED_VERSIONS: Final[TypeAdapter[tuple[MCPLegacyVersion, ...]]] = TypeAdapter(MCPAdvertisedVersions)


def configured_versions() -> tuple[str, ...]:
    from litellm.proxy.proxy_server import general_settings_view

    configured: Final = general_settings_view().get("mcp_advertised_versions")
    return _ADVERTISED_VERSIONS.validate_python(MCP_LEGACY_VERSIONS if configured is None else configured)


def build_discovery(
    *,
    configured: tuple[str, ...],
    revision: str,
    transport: MCPTransport,
    authorized_operations: frozenset[str],
    upstream_versions: frozenset[str],
    capabilities: ServerCapabilities,
    client_extensions: frozenset[str] = frozenset(),
    upstream_extensions: frozenset[str] = frozenset(),
    instructions: str | None = None,
) -> DiscoverResult:
    supported: Final = tuple(
        version
        for version, support in REVISION_SUPPORT.items()
        if version in configured and support.completed and transport in support.transports
    )
    revision_support: Final = REVISION_SUPPORT.get(revision)
    operations: Final[frozenset[str]] = (
        authorized_operations & revision_support.operations
        if revision in supported
        and revision_support is not None
        and any((revision, upstream) in TRANSLATION_PAIRS for upstream in upstream_versions)
        else frozenset()
    )
    extensions: Final[frozenset[str]] = (
        revision_support.extensions & client_extensions & upstream_extensions
        if operations and revision_support is not None
        else frozenset()
    )
    caller_capabilities: Final = capabilities.model_copy(deep=True)
    return DiscoverResult(
        supported_versions=list(supported),
        capabilities=ServerCapabilities(
            tools=caller_capabilities.tools if {"tools/list", "tools/call"} <= operations else None,
            prompts=caller_capabilities.prompts if {"prompts/list", "prompts/get"} <= operations else None,
            resources=caller_capabilities.resources if {"resources/list", "resources/read"} <= operations else None,
            extensions={
                key: value for key, value in (caller_capabilities.extensions or {}).items() if key in extensions
            }
            or None,
        ),
        instructions=instructions,
        cache_scope="private",
        ttl_ms=0,
    )


class GatewayVersionPolicy:
    def __init__(self, versions: Callable[[], tuple[str, ...]] = configured_versions) -> None:
        self._versions = versions

    async def __call__(self, ctx: ServerRequestContext[object, object], call_next: CallNext) -> HandlerResult:
        versions: Final = self._versions()
        requested: Final = (
            InitializeRequestParams.model_validate(ctx.params or {}).protocol_version
            if ctx.method == "initialize"
            else ctx.protocol_version
        )
        negotiated: Final = (
            (requested if requested in HANDSHAKE_PROTOCOL_VERSIONS else LATEST_HANDSHAKE_VERSION)
            if ctx.method == "initialize"
            else requested
        )
        if negotiated not in versions:
            raise MCPError(code=-32022, message="Unsupported MCP protocol version", data={"supported": list(versions)})
        result: Final = await call_next(ctx)
        if ctx.method != "initialize":
            return result
        initialized: Final = InitializeResult.model_validate(result)
        discovery: Final = build_discovery(
            configured=versions,
            revision=initialized.protocol_version,
            transport=MCPTransport.http,
            authorized_operations=GATEWAY_OPERATIONS,
            upstream_versions=frozenset(HANDSHAKE_PROTOCOL_VERSIONS),
            capabilities=initialized.capabilities,
            instructions=initialized.instructions,
        )
        return initialized.model_copy(update={"capabilities": discovery.capabilities})
