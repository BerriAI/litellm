"""
Tests for Dominion Observatory MCP trust scoring.
"""

import httpx
import pytest

from litellm.proxy._experimental.mcp_server.mcp_trust_scoring import (
    MCPTrustScoringClient,
    MCPTrustScoringConfig,
    assert_requested_server_passes_trust_filter,
    filter_mcp_servers_by_trust,
    initialize_mcp_trust_scoring_from_config,
    normalize_trust_score,
    parse_trust_response,
    rank_mcp_servers_by_trust,
    sanitize_url_for_trust_lookup,
)
from litellm.proxy._types import MCPTransport
from litellm.types.mcp import MCPTransportType
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def _server(
    *,
    server_id: str = "srv-1",
    name: str = "alpha",
    url: str | None = "https://alpha.example.com/mcp",
    transport: MCPTransportType = MCPTransport.http,
) -> MCPServer:
    return MCPServer(
        server_id=server_id,
        name=name,
        url=url,
        transport=transport,
    )


def _trust_response(
    *,
    status_code: int = 200,
    payload: object | None = None,
) -> httpx.Response:
    request = httpx.Request("GET", "https://dominionobservatory.com/api/trust")
    return httpx.Response(
        status_code=status_code,
        request=request,
        json=payload,
    )


class _FakeHttpClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    async def get(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        del headers
        self.calls.append((url, params))
        if not self._responses:
            raise RuntimeError("no mocked responses left")
        return self._responses.pop(0)


class _FakeSharedCache:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    async def async_get_cache(
        self,
        key: str,
        local_only: bool = False,
        **kwargs: object,
    ) -> object | None:
        del local_only, kwargs
        return self.store.get(key)

    async def async_set_cache(
        self,
        key: str,
        value: object,
        local_only: bool = False,
        **kwargs: object,
    ) -> None:
        del local_only, kwargs
        self.store[key] = value


def _isolated_client(
    config: MCPTrustScoringConfig,
    http_client: _FakeHttpClient,
) -> MCPTrustScoringClient:
    shared_cache = _FakeSharedCache()

    def _cache_getter() -> _FakeSharedCache:
        return shared_cache

    def _http_client_getter() -> _FakeHttpClient:
        return http_client

    return MCPTrustScoringClient(
        config,
        cache_getter=_cache_getter,
        http_client_getter=_http_client_getter,
    )


def test_normalize_trust_score_maps_do_percent_scale_to_fraction() -> None:
    assert normalize_trust_score(85) == 0.85
    normalized = normalize_trust_score(92.5)
    assert normalized is not None
    assert abs(normalized - 0.925) < 1e-9
    low_trust = normalize_trust_score(1.0)
    assert low_trust is not None
    assert abs(low_trust - 0.01) < 1e-9


def test_sanitize_url_for_trust_lookup_strips_userinfo_and_sensitive_query_params() -> (
    None
):
    sanitized = sanitize_url_for_trust_lookup(
        "https://secret-token@mcp.example.com:8443/mcp?api_key=leak&safe=1#frag"
    )

    assert sanitized == "https://mcp.example.com:8443/mcp?safe=1"


def test_assert_requested_server_passes_trust_filter_rejects_untrusted_server() -> None:
    from fastapi import HTTPException

    trusted = _server(server_id="srv-trusted")
    with pytest.raises(HTTPException) as exc_info:
        assert_requested_server_passes_trust_filter(
            filtered_servers=[trusted],
            server_id="srv-untrusted",
        )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_get_trust_score_forwards_sanitized_url_to_do() -> None:
    http_client = _FakeHttpClient(
        [
            _trust_response(
                payload={
                    "found": True,
                    "server_url": "https://mcp.example.com/mcp",
                    "trust_score": 90,
                }
            )
        ]
    )
    config = MCPTrustScoringConfig(enabled=True)
    client = _isolated_client(
        config,
        http_client,
    )

    await client.get_trust_score(
        "https://secret@mcp.example.com/mcp?api_key=leak&safe=1"
    )

    assert http_client.calls == [
        (
            config.api_url,
            {"url": "https://mcp.example.com/mcp?safe=1"},
        )
    ]


def test_parse_trust_response_found_server() -> None:
    result = parse_trust_response(
        server_url="https://alpha.example.com/mcp",
        payload={
            "found": True,
            "server_url": "https://alpha.example.com/mcp",
            "trust_score": 92.5,
        },
    )

    assert result.found is True
    assert result.trust_score_raw == 92.5
    assert result.trust_score_normalized is not None
    assert abs(result.trust_score_normalized - 0.925) < 1e-9
    assert result.lookup_status == "ok"


def test_initialize_from_config_disabled() -> None:
    client = initialize_mcp_trust_scoring_from_config({"enabled": False})
    assert client is None


@pytest.mark.asyncio
async def test_get_trust_score_fetches_and_caches() -> None:
    http_client = _FakeHttpClient(
        [
            _trust_response(
                payload={
                    "found": True,
                    "server_url": "https://cache-test.example.com/mcp",
                    "trust_score": 80,
                }
            )
        ]
    )
    client = _isolated_client(
        MCPTrustScoringConfig(enabled=True),
        http_client,
    )

    first = await client.get_trust_score("https://cache-test.example.com/mcp")
    second = await client.get_trust_score("https://cache-test.example.com/mcp")

    assert first.trust_score_normalized is not None
    assert abs(first.trust_score_normalized - 0.8) < 1e-9
    assert second == first
    assert len(http_client.calls) == 1


@pytest.mark.asyncio
async def test_filter_servers_by_trust_excludes_low_score() -> None:
    http_client = _FakeHttpClient(
        [
            _trust_response(
                payload={
                    "found": True,
                    "server_url": "https://alpha.example.com/mcp",
                    "trust_score": 90,
                }
            ),
            _trust_response(
                payload={
                    "found": True,
                    "server_url": "https://beta.example.com/mcp",
                    "trust_score": 20,
                }
            ),
        ]
    )
    client = _isolated_client(
        MCPTrustScoringConfig(enabled=True, min_trust_score=0.5),
        http_client,
    )

    filtered = await client.filter_servers_by_trust(
        [
            _server(
                server_id="srv-1", name="alpha", url="https://alpha.example.com/mcp"
            ),
            _server(server_id="srv-2", name="beta", url="https://beta.example.com/mcp"),
        ]
    )

    assert [server.server_id for server in filtered] == ["srv-1"]


@pytest.mark.asyncio
async def test_filter_servers_fail_open_on_lookup_error() -> None:
    http_client = _FakeHttpClient(
        [_trust_response(status_code=503, payload={"error": "busy"})]
    )
    client = _isolated_client(
        MCPTrustScoringConfig(enabled=True, fail_open=True),
        http_client,
    )

    filtered = await client.filter_servers_by_trust(
        [_server(url="https://fail-open.example.com/mcp")]
    )

    assert len(filtered) == 1


@pytest.mark.asyncio
async def test_filter_servers_fail_closed_on_lookup_error() -> None:
    http_client = _FakeHttpClient(
        [_trust_response(status_code=503, payload={"error": "busy"})]
    )
    client = _isolated_client(
        MCPTrustScoringConfig(enabled=True, fail_open=False),
        http_client,
    )

    filtered = await client.filter_servers_by_trust(
        [_server(url="https://fail-closed.example.com/mcp")]
    )

    assert filtered == ()


@pytest.mark.asyncio
async def test_stdio_server_is_skipped_not_filtered() -> None:
    client = _isolated_client(
        MCPTrustScoringConfig(enabled=True, min_trust_score=0.99),
        _FakeHttpClient([]),
    )

    filtered = await client.filter_servers_by_trust(
        [_server(server_id="stdio-1", url=None, transport=MCPTransport.stdio)]
    )

    assert [server.server_id for server in filtered] == ["stdio-1"]


@pytest.mark.asyncio
async def test_rank_servers_by_trust_orders_highest_first() -> None:
    http_client = _FakeHttpClient(
        [
            _trust_response(
                payload={
                    "found": True,
                    "server_url": "https://beta.example.com/mcp",
                    "trust_score": 40,
                }
            ),
            _trust_response(
                payload={
                    "found": True,
                    "server_url": "https://alpha.example.com/mcp",
                    "trust_score": 95,
                }
            ),
        ]
    )
    client = _isolated_client(
        MCPTrustScoringConfig(enabled=True),
        http_client,
    )

    ranked = await client.rank_servers_by_trust(
        [
            _server(server_id="srv-2", name="beta", url="https://beta.example.com/mcp"),
            _server(
                server_id="srv-1", name="alpha", url="https://alpha.example.com/mcp"
            ),
        ]
    )

    assert [server.server_id for server in ranked] == ["srv-1", "srv-2"]


@pytest.mark.asyncio
async def test_module_helpers_use_explicit_client() -> None:
    http_client = _FakeHttpClient(
        [
            _trust_response(
                payload={
                    "found": True,
                    "server_url": "https://alpha.example.com/mcp",
                    "trust_score": 95,
                }
            )
        ]
    )
    client = _isolated_client(
        MCPTrustScoringConfig(enabled=True, min_trust_score=0.5),
        http_client,
    )

    filtered = await filter_mcp_servers_by_trust(
        [_server(url="https://helper-test.example.com/mcp")],
        client=client,
    )
    ranked = await rank_mcp_servers_by_trust(
        [_server(url="https://helper-test.example.com/mcp")],
        client=client,
    )

    assert len(filtered) == 1
    assert len(ranked) == 1


@pytest.mark.asyncio
async def test_manager_list_tools_applies_trust_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )

    http_client = _FakeHttpClient(
        [
            _trust_response(
                payload={
                    "found": True,
                    "server_url": "https://alpha.example.com/mcp",
                    "trust_score": 90,
                }
            ),
            _trust_response(
                payload={
                    "found": True,
                    "server_url": "https://beta.example.com/mcp",
                    "trust_score": 20,
                }
            ),
        ]
    )
    client = _isolated_client(
        MCPTrustScoringConfig(enabled=True, min_trust_score=0.5),
        http_client,
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_trust_scoring.get_mcp_trust_scoring_client",
        lambda: client,
    )

    manager = MCPServerManager()
    alpha = _server(
        server_id="srv-1", name="alpha", url="https://alpha.example.com/mcp"
    )
    beta = _server(server_id="srv-2", name="beta", url="https://beta.example.com/mcp")
    manager.registry = {"srv-1": alpha, "srv-2": beta}

    async def _fake_get_allowed(_user_api_key_auth: object | None = None) -> list[str]:
        del _user_api_key_auth
        return ["srv-1", "srv-2"]

    async def _fake_get_tools(*, server: MCPServer, **kwargs: object) -> list[object]:
        del kwargs
        fetched_server_ids.append(server.server_id)
        return []

    fetched_server_ids: list[str] = []
    monkeypatch.setattr(manager, "get_allowed_mcp_servers", _fake_get_allowed)
    monkeypatch.setattr(manager, "_get_tools_from_server", _fake_get_tools)

    await manager.list_tools()

    assert fetched_server_ids == ["srv-1"]
