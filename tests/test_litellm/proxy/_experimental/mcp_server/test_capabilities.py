from typing import Final

import pytest
from mcp import Client
from mcp.server import Server
from mcp.shared.exceptions import MCPError
from mcp.types import PromptsCapability, ResourcesCapability, ServerCapabilities, ToolsCapability
from mcp_types.version import HANDSHAKE_PROTOCOL_VERSIONS

from litellm.proxy._experimental.mcp_server.capabilities import (
    GATEWAY_OPERATIONS,
    REVISION_SUPPORT,
    TRANSLATION_PAIRS,
    GatewayVersionPolicy,
    build_discovery,
)
from litellm.types.mcp import MCPTransport


@pytest.mark.parametrize("revision", HANDSHAKE_PROTOCOL_VERSIONS)
@pytest.mark.parametrize("transport", tuple(MCPTransport))
def test_discovery_only_exposes_authorized_completed_support(revision, transport):
    result = build_discovery(
        configured=(revision, "2026-07-28", "unknown"),
        revision=revision,
        transport=transport,
        authorized_operations=frozenset({"tools/list", "tools/call"}),
        upstream_versions=frozenset(HANDSHAKE_PROTOCOL_VERSIONS),
        capabilities=ServerCapabilities(
            tools=ToolsCapability(), prompts=PromptsCapability(), resources=ResourcesCapability(),
            extensions={"io.modelcontextprotocol/ui": {}},
        ),
        client_extensions=frozenset({"io.modelcontextprotocol/ui"}),
        upstream_extensions=frozenset({"io.modelcontextprotocol/ui"}),
    )
    assert result.supported_versions == [revision]
    assert result.capabilities.tools is not None
    assert result.capabilities.prompts is None
    assert result.capabilities.resources is None
    assert result.capabilities.extensions is None
    assert result.capabilities.tasks is None
    assert result.cache_scope == "private"
    assert result.ttl_ms == 0


@pytest.mark.parametrize("upstream", [frozenset(), frozenset({"unknown"}), frozenset({"2026-07-28"})])
def test_unproven_translation_never_advertises_operations(upstream):
    result = build_discovery(
        configured=HANDSHAKE_PROTOCOL_VERSIONS,
        revision="2025-11-25",
        transport=MCPTransport.http,
        authorized_operations=GATEWAY_OPERATIONS,
        upstream_versions=upstream,
        capabilities=ServerCapabilities(tools=ToolsCapability()),
    )
    assert result.capabilities.tools is None


@pytest.mark.parametrize("revision", ["2026-07-28", "unknown", "2024-11-05"])
def test_unadvertised_revision_never_gains_capabilities(revision):
    result = build_discovery(
        configured=("2025-11-25",), revision=revision, transport=MCPTransport.http,
        authorized_operations=GATEWAY_OPERATIONS, upstream_versions=frozenset(HANDSHAKE_PROTOCOL_VERSIONS),
        capabilities=ServerCapabilities(tools=ToolsCapability()),
    )
    assert result.capabilities.tools is None


def test_discovery_results_do_not_share_mutable_capabilities():
    capabilities = ServerCapabilities(tools=ToolsCapability(), prompts=PromptsCapability(), resources=ResourcesCapability())
    args = dict(
        configured=HANDSHAKE_PROTOCOL_VERSIONS, revision="2025-11-25", transport=MCPTransport.http,
        upstream_versions=frozenset(HANDSHAKE_PROTOCOL_VERSIONS), capabilities=capabilities,
    )
    allowed = build_discovery(**args, authorized_operations=GATEWAY_OPERATIONS)
    denied = build_discovery(**args, authorized_operations=frozenset())
    assert allowed.capabilities.prompts is not None
    assert allowed.capabilities.resources is not None
    assert denied.capabilities.model_dump(exclude_none=True) == {}
    assert allowed.capabilities.tools is not None
    allowed.capabilities.tools.list_changed = True
    assert capabilities.tools.list_changed is not True


def test_modern_candidates_do_not_enable_public_serving():
    modern = REVISION_SUPPORT["2026-07-28"]
    assert modern.completed is False
    assert "input_required" in modern.results
    assert MCPTransport.sse not in modern.transports
    assert not any("2026-07-28" in pair for pair in TRANSLATION_PAIRS)


@pytest.mark.asyncio
@pytest.mark.parametrize("versions,accepted", [(("2025-11-25",), True), (("2025-06-18",), False)])
async def test_version_policy_gates_the_actual_sdk_handshake(versions, accepted):
    server: Final = Server("test-gateway", version="1")
    server.middleware.append(GatewayVersionPolicy(lambda: versions))
    if accepted:
        async with Client(server, mode="legacy") as client:
            assert client.protocol_version == "2025-11-25"
            result = await client.session.send_ping()
            assert result is not None
    else:
        with pytest.RaisesGroup(pytest.RaisesExc(MCPError, match="Unsupported MCP protocol version"), flatten_subgroups=True):
            async with Client(server, mode="legacy"):
                pytest.fail("The excluded revision must not initialize")
