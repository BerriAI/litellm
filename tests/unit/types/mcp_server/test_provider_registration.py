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
