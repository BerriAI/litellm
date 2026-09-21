"""Tests for the shared MCP header primitives.

``same_header`` / ``has_header`` / ``without_header`` are the one owner of "is this the credential's
header", used by both MCP stacks and the upstream-credential resolver. They live here rather than in
either stack because a second implementation is exactly how an injected header came to shadow a
resolved credential on one path and not the other.
"""

import pytest

from litellm.types.mcp import (
    credential_redirect_hook,
    crosses_origin,
    has_header,
    same_header,
    without_header,
)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("Authorization", "authorization", True),
        ("ESB-OAuth", "esb-oauth", True),
        ("esb-oauth", "esb-oauth", True),
        ("esb-oauth", "esb_oauth", False),
        ("esb-oauth", "Authorization", False),
    ],
)
def test_header_names_compare_case_insensitively(a: str, b: str, expected: bool) -> None:
    # RFC 7230 3.2. Every consumer of a credential slot routes through this, so a case-sensitive
    # comparison anywhere would let an injected header shadow a resolved credential.
    assert same_header(a, b) is expected


def test_without_header_drops_every_casing_and_keeps_the_rest() -> None:
    headers = {"ESB-OAuth": "injected", "esb-oauth": "also injected", "X-Trace": "keep"}
    assert without_header(headers, "esb-oauth") == {"X-Trace": "keep"}


def test_without_header_collapses_to_none_when_nothing_remains() -> None:
    assert without_header({"Authorization": "Bearer x"}, "AUTHORIZATION") is None
    assert without_header(None, "esb-oauth") is None
    assert without_header({}, "esb-oauth") is None


def test_has_header_matches_any_casing() -> None:
    assert has_header({"ESB-OAuth": "v"}, "esb-oauth") is True
    assert has_header({"X-Other": "v"}, "esb-oauth") is False
    assert has_header(None, "esb-oauth") is False


@pytest.mark.parametrize(
    "target,expected",
    [
        ("https://upstream.example.com/other", False),      # same origin
        ("https://upstream.example.com:443/other", False),  # explicit default port
        ("https://attacker.example.com/collect", True),     # different host
        ("http://upstream.example.com/collect", True),      # scheme downgrade, same host
        ("https://upstream.example.com:8443/other", True),  # different port, same host
        ("https://sub.upstream.example.com/x", True),       # different host
    ],
)
def test_origin_is_scheme_host_and_port_not_host_alone(target: str, expected: bool) -> None:
    assert crosses_origin("https://upstream.example.com/mcp", target) is expected


def test_an_https_upgrade_of_the_same_host_is_not_crossing() -> None:
    # HTTP clients exempt this when deciding to keep Authorization, so a credential slot that did
    # not would lose the credential on every such redirect.
    assert crosses_origin("http://upstream.example.com/mcp", "https://upstream.example.com/x") is False
    assert crosses_origin("http://upstream.example.com/mcp", "http://upstream.example.com/x") is False


@pytest.mark.asyncio
async def test_the_hook_drops_the_slot_only_once_the_origin_changes() -> None:
    import httpx

    hook = credential_redirect_hook("https://upstream.example.com/mcp", "esb-oauth")

    same = httpx.Request("GET", "https://upstream.example.com/other", headers={"esb-oauth": "Bearer x"})
    await hook(same)
    assert same.headers["esb-oauth"] == "Bearer x"

    foreign = httpx.Request("GET", "https://attacker.example.com/x", headers={"esb-oauth": "Bearer x"})
    await hook(foreign)
    assert "esb-oauth" not in foreign.headers
