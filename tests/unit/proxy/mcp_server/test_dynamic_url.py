from typing import Final

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from litellm.proxy.mcp_server.dynamic_url import (
    DynamicURLAccepted,
    DynamicURLMissing,
    DynamicURLRejected,
    resolve_dynamic_url,
    server_for_request,
)
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPDynamicURL, MCPServer

_POLICY: Final = MCPDynamicURL(allowed_hosts_regex=(r"[a-z0-9-]+\.warehouse\.example",))


def _warehouse(dynamic_url: MCPDynamicURL | None = _POLICY, url: str | None = None) -> MCPServer:
    return MCPServer(
        server_id="warehouse",
        name="warehouse",
        transport=MCPTransport.http,
        url=url,
        dynamic_url=dynamic_url,
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://acct1.warehouse.example/mcp",
        "https://acct1.warehouse.example/mcp?key=tenant-secret",
        "https://acct-2.warehouse.example:8443/mcp",
        "HTTPS://ACCT1.WAREHOUSE.EXAMPLE/mcp",
    ],
)
def test_an_allowed_url_is_used_exactly_as_sent(url: str) -> None:
    assert resolve_dynamic_url(_POLICY, {"X-LiteLLM-MCP-URL": url}) == DynamicURLAccepted(url=url)


@pytest.mark.parametrize(
    "url",
    [
        "https://warehouse.example/mcp",
        "https://acct1.warehouse.example.attacker.test/mcp",
        "https://attacker.test/acct1.warehouse.example",
        "https://a.b.warehouse.example/mcp",
        "https://acct1.warehouse.example./mcp",
        "http://acct1.warehouse.example/mcp",
        "ftp://acct1.warehouse.example/mcp",
        "file:///etc/passwd",
        "acct1.warehouse.example/mcp",
        "https://user:pass@acct1.warehouse.example/mcp",
        "https://user@acct1.warehouse.example/mcp",
        "https://attacker.test\\@acct1.warehouse.example/mcp",
        "https://acct1.warehouse.example/mcp#fragment",
        "https://acct1.warehouse.example/mcp#",
        "https://acct1.warehouse.example:notaport/mcp",
        f"https://{'a' * 250}.warehouse.example/mcp",
    ],
)
def test_a_url_outside_the_policy_is_rejected(url: str) -> None:
    assert resolve_dynamic_url(_POLICY, {"x-litellm-mcp-url": url}) == DynamicURLRejected()


def test_http_is_accepted_only_when_the_admin_allows_it() -> None:
    http_policy: Final = MCPDynamicURL(allowed_hosts_regex=_POLICY.allowed_hosts_regex, allow_http=True)
    url: Final = "http://acct-2.warehouse.example:8443/mcp"

    assert resolve_dynamic_url(http_policy, {"x-litellm-mcp-url": url}) == DynamicURLAccepted(url=url)


@pytest.mark.parametrize("headers", [None, {}, {"x-litellm-mcp-url": ""}, {"x-mcp-url": "https://a.warehouse.example"}])
def test_no_header_is_missing_not_rejected(headers: dict[str, str] | None) -> None:
    assert resolve_dynamic_url(_POLICY, headers) == DynamicURLMissing()


def test_a_pattern_that_backtracks_forever_rejects_instead_of_hanging() -> None:
    policy: Final = MCPDynamicURL(allowed_hosts_regex=(r"(a|aa)+",))

    result: Final = resolve_dynamic_url(policy, {"x-litellm-mcp-url": f"https://{'a' * 80}-/mcp"})

    assert result == DynamicURLRejected()


def test_the_request_server_carries_the_header_url_and_the_registered_one_is_untouched() -> None:
    registered: Final = _warehouse()

    tenant_a: Final = server_for_request(registered, {"x-litellm-mcp-url": "https://a.warehouse.example/mcp"})
    tenant_b: Final = server_for_request(registered, {"x-litellm-mcp-url": "https://b.warehouse.example/mcp"})

    assert (tenant_a.url, tenant_b.url, registered.url) == (
        "https://a.warehouse.example/mcp",
        "https://b.warehouse.example/mcp",
        None,
    )


def test_a_server_without_dynamic_url_ignores_the_header() -> None:
    fixed: Final = _warehouse(dynamic_url=None, url="https://fixed.example/mcp")

    assert server_for_request(fixed, {"x-litellm-mcp-url": "https://a.warehouse.example/mcp"}) is fixed


@pytest.mark.parametrize(
    ("headers", "detail_fragment"),
    [
        (None, "requires the x-litellm-mcp-url header"),
        ({"x-litellm-mcp-url": "https://attacker.test/mcp?key=tenant-secret"}, "not an allowed URL"),
    ],
    ids=["missing", "rejected"],
)
def test_a_missing_or_rejected_header_is_a_400_that_does_not_echo_the_url(
    headers: dict[str, str] | None, detail_fragment: str
) -> None:
    with pytest.raises(HTTPException) as error:
        server_for_request(_warehouse(), headers)

    assert error.value.status_code == 400
    assert detail_fragment in str(error.value.detail)
    assert "attacker.test" not in str(error.value.detail)
    assert "tenant-secret" not in str(error.value.detail)


@pytest.mark.parametrize("patterns", [(), ("",), ("[a-z",)], ids=["empty", "blank", "invalid"])
def test_the_policy_needs_at_least_one_compiling_pattern(patterns: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        MCPDynamicURL(allowed_hosts_regex=patterns)


def test_a_request_supplied_url_does_not_follow_redirects_and_logs_only_its_origin() -> None:
    from litellm.mcp_client.client import MCPClient

    url: Final = "https://acct1.warehouse.example:8443/t/tenant-secret/mcp?key=tenant-secret"
    dynamic: Final = MCPClient(server_url=url, request_supplied_url=True)
    configured: Final = MCPClient(server_url=url)

    assert dynamic._create_httpx_client_factory()().follow_redirects is False
    assert dynamic._log_target == "https://acct1.warehouse.example:8443"
    assert configured._create_httpx_client_factory()().follow_redirects is True
    assert configured._log_target == url


def test_the_url_header_never_reaches_logging_metadata() -> None:
    from litellm.proxy.mcp_server.utils import logging_safe_mcp_headers

    headers: Final = {"X-LiteLLM-MCP-URL": "https://acct1.warehouse.example/mcp?key=tenant-secret", "x-trace": "1"}

    assert logging_safe_mcp_headers(headers) == {"x-trace": "1"}
