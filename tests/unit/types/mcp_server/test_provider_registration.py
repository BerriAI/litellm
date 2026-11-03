from typing import Final

import pytest

from litellm.proxy._types import NewMCPServerRequest
from litellm.proxy.mcp_server.db import _prepare_mcp_server_data
from litellm.types.mcp_server.provider_registration import canonical_provider_mcp_url, normalize_provider_registration


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://DOCS.example:443/mcp/tenant?scope=a", "https://docs.example/mcp/tenant?scope=a"),
        ("https://docs.example:8443/mcp/", "https://docs.example:8443/mcp/"),
        ("https://[::1]:443/mcp", "https://[::1]/mcp"),
    ],
)
def test_endpoint_identity_preserves_path_query_and_nondefault_port(url: str, expected: str) -> None:
    assert canonical_provider_mcp_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://docs.example/mcp",
        "https://u:p@docs.example/mcp",
        "https://docs.example/mcp#x",
        "https://docs.example/mcp\n",
    ],
)
def test_unsafe_endpoint_identity_is_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="Provider MCP url"):
        canonical_provider_mcp_url(url)


@pytest.mark.parametrize(
    "patch",
    [
        {"allowed_tools": None},
        {"allowed_tools": ["*"]},
        {"allowed_tools": [" "]},
        {"transport": "http"},
        {"auth_type": "none"},
        {"credentials": {"auth_value": "synthetic"}},
        {"static_headers": {"Authorization": "synthetic"}},
        {"command": "python"},
        {"gateway_custom_option": {"enabled": True}},
        {"timeout": 0},
        {"mcp_info": {"tool_allowlist_enforced": False}},
        {"mcp_info": {"tool_allowlist_enforced": None}},
        {"tool_name_to_description": {"not_allowed": "Not in the list"}},
    ],
)
def test_provider_registration_rejects_unbounded_tools_and_connection_settings(patch: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"allowed_tools|valid list|Provider|Tool descriptions"):
        normalize_provider_registration({"url": "https://docs.example/mcp", "allowed_tools": ["search"], **patch})


def test_empty_tool_list_is_enforced_without_transport_or_stored_credentials() -> None:
    request: Final = NewMCPServerRequest(
        execution_mode="provider", url="https://docs.example/mcp", allowed_tools=[], mcp_info={"description": "Docs"}
    )
    prepared: Final = _prepare_mcp_server_data(request)
    assert prepared["allowed_tools"] == []
    assert request.mcp_info == {"description": "Docs", "tool_allowlist_enforced": True}
    assert prepared["transport"] is None
    assert not prepared.get("credentials")
    assert prepared.get("auth_type") is None


@pytest.mark.parametrize("neutral", [None, False, "", {}, []])
def test_provider_registration_ignores_neutral_unknown_settings(neutral: object) -> None:
    request: Final = NewMCPServerRequest(
        execution_mode="provider", url="https://docs.example/mcp", allowed_tools=["search"],
        gateway_custom_option=neutral,
    )
    assert request.allowed_tools == ["search"]
    assert request.transport is None
    assert "gateway_custom_option" not in request.model_dump()


def test_provider_registration_preserves_supported_management_fields() -> None:
    request: Final = NewMCPServerRequest(
        execution_mode="provider", server_name="docs", alias="documentation",
        url="https://docs.example/mcp", allowed_tools=["search"], description="Documentation",
        instructions="Search docs", source_url="https://docs.example", mcp_access_groups=["research"],
        allow_all_keys=True, tool_name_to_description={"search": "Search docs"},
        tool_name_to_display_name={"search": "Search"}, mcp_info={"custom_metadata": "retained"},
    )
    assert request.server_name == "docs" and request.alias == "documentation"
    assert request.description == "Documentation" and request.instructions == "Search docs"
    assert request.source_url == "https://docs.example" and request.mcp_access_groups == ["research"]
    assert request.allow_all_keys
    assert request.tool_name_to_description == {"search": "Search docs"}
    assert request.tool_name_to_display_name == {"search": "Search"}
    assert request.mcp_info == {"custom_metadata": "retained", "tool_allowlist_enforced": True}
