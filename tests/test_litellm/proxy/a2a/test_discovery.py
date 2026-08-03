"""Tests for the well-known card fetcher and the discovery endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.a2a.discovery import (
    AGENT_CARD_WELL_KNOWN_PATHS,
    AgentCardDiscoveryError,
    DiscoveryMode,
    fetch_well_known_card,
)
from litellm.proxy.a2a.endpoints import router as a2a_router
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


@pytest.fixture(autouse=True)
def _disable_url_validation_for_mocks(monkeypatch):
    """The fetch tests use placeholder hostnames (``upstream.example``,
    ``localhost:2024``) with mocked HTTP clients. ``async_safe_get`` would
    otherwise resolve those hostnames and either fail DNS or block on the
    SSRF guard. Disabling validation here lets the unit tests focus on
    fallback / parsing logic; SSRF behavior is covered in its own test."""
    monkeypatch.setattr(litellm, "user_url_validation", False)


# ---------------------------------------------------------------------------
# fetch_well_known_card
# ---------------------------------------------------------------------------


def _mock_response(status_code: int = 200, body=None, raise_json=False):
    response = MagicMock()
    response.status_code = status_code
    if raise_json:
        response.json = MagicMock(side_effect=ValueError("bad json"))
    else:
        response.json = MagicMock(return_value=body)
    return response


@pytest.mark.asyncio
async def test_fetch_uses_first_path_that_returns_200():
    body = {"name": "agent"}
    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=_mock_response(200, body=body))

    with patch(
        "litellm.proxy.a2a.discovery.get_async_httpx_client", return_value=fake_client
    ):
        card = await fetch_well_known_card("https://upstream.example")

    assert card == body
    # First call should be to the canonical path.
    called_url = fake_client.get.call_args.args[0]
    assert called_url == f"https://upstream.example{AGENT_CARD_WELL_KNOWN_PATHS[0]}"


@pytest.mark.asyncio
async def test_fetch_falls_back_to_later_paths_on_404():
    body = {"name": "agent"}
    fake_client = MagicMock()
    fake_client.get = AsyncMock(
        side_effect=[
            _mock_response(404),
            _mock_response(404),
            _mock_response(200, body=body),
        ]
    )

    with patch(
        "litellm.proxy.a2a.discovery.get_async_httpx_client", return_value=fake_client
    ):
        card = await fetch_well_known_card("https://upstream.example")

    assert card == body
    assert fake_client.get.await_count == len(AGENT_CARD_WELL_KNOWN_PATHS)


@pytest.mark.asyncio
async def test_fetch_raises_when_all_paths_fail():
    fake_client = MagicMock()
    fake_client.get = AsyncMock(
        side_effect=[_mock_response(404) for _ in AGENT_CARD_WELL_KNOWN_PATHS]
    )

    with patch(
        "litellm.proxy.a2a.discovery.get_async_httpx_client", return_value=fake_client
    ):
        with pytest.raises(AgentCardDiscoveryError):
            await fetch_well_known_card("https://upstream.example")


@pytest.mark.asyncio
async def test_fetch_skips_path_that_returns_non_json_body():
    body = {"name": "agent"}
    fake_client = MagicMock()
    fake_client.get = AsyncMock(
        side_effect=[
            _mock_response(200, raise_json=True),
            _mock_response(200, body=body),
        ]
    )

    with patch(
        "litellm.proxy.a2a.discovery.get_async_httpx_client", return_value=fake_client
    ):
        card = await fetch_well_known_card("https://upstream.example")

    assert card == body


@pytest.mark.asyncio
async def test_fetch_skips_path_that_returns_non_object_json():
    fake_client = MagicMock()
    fake_client.get = AsyncMock(
        side_effect=[
            _mock_response(200, body=["not", "an", "object"]),
            _mock_response(200, body={"name": "agent"}),
            _mock_response(404),
        ]
    )

    with patch(
        "litellm.proxy.a2a.discovery.get_async_httpx_client", return_value=fake_client
    ):
        card = await fetch_well_known_card("https://upstream.example")

    assert card == {"name": "agent"}


@pytest.mark.asyncio
async def test_fetch_requires_base_url():
    with pytest.raises(AgentCardDiscoveryError):
        await fetch_well_known_card("")


# ---------------------------------------------------------------------------
# LangGraph Platform discovery mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_langgraph_mode_appends_assistant_id_query_param():
    """LangGraph serves one card endpoint; the assistant is selected via query string."""
    body = {"name": "support-agent"}
    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=_mock_response(200, body=body))

    with patch(
        "litellm.proxy.a2a.discovery.get_async_httpx_client", return_value=fake_client
    ):
        card = await fetch_well_known_card(
            "http://localhost:2024",
            discovery_mode=DiscoveryMode.LANGGRAPH_PLATFORM,
            params={"assistant_id": "agent"},
        )

    assert card == body
    called_url = fake_client.get.call_args.args[0]
    # The canonical A2A path with the LangGraph query parameter — NOT a
    # per-assistant subpath like /agent/.well-known/agent-card.json.
    assert called_url == (
        "http://localhost:2024/.well-known/agent-card.json?assistant_id=agent"
    )


@pytest.mark.asyncio
async def test_langgraph_mode_requires_assistant_id():
    with pytest.raises(AgentCardDiscoveryError, match="assistant_id"):
        await fetch_well_known_card(
            "http://localhost:2024",
            discovery_mode=DiscoveryMode.LANGGRAPH_PLATFORM,
            params={},
        )


@pytest.mark.asyncio
async def test_langgraph_mode_falls_back_to_older_well_known_paths():
    """If an older LangGraph deployment serves /.well-known/agent.json, accept that too."""
    fake_client = MagicMock()
    fake_client.get = AsyncMock(
        side_effect=[
            _mock_response(404),
            _mock_response(200, body={"name": "support-agent"}),
        ]
    )

    with patch(
        "litellm.proxy.a2a.discovery.get_async_httpx_client", return_value=fake_client
    ):
        card = await fetch_well_known_card(
            "http://localhost:2024",
            discovery_mode=DiscoveryMode.LANGGRAPH_PLATFORM,
            params={"assistant_id": "agent"},
        )

    assert card == {"name": "support-agent"}
    # Both calls carry the assistant_id query param.
    for call in fake_client.get.await_args_list:
        assert "assistant_id=agent" in call.args[0]


# ---------------------------------------------------------------------------
# POST /v1/a2a/discover
# ---------------------------------------------------------------------------


def _client_for_role(role: LitellmUserRoles) -> TestClient:
    app = FastAPI()
    app.include_router(a2a_router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="u", user_role=role
    )
    return TestClient(app)


def test_discover_admin_returns_raw_card():
    client = _client_for_role(LitellmUserRoles.PROXY_ADMIN)
    with patch(
        "litellm.proxy.a2a.endpoints.fetch_well_known_card",
        new=AsyncMock(return_value={"name": "Upstream"}),
    ):
        resp = client.post("/v1/a2a/discover", json={"url": "https://upstream.example"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["url"] == "https://upstream.example"
    assert body["agent_card"] == {"name": "Upstream"}


def test_discover_non_admin_forbidden():
    client = _client_for_role(LitellmUserRoles.INTERNAL_USER)
    resp = client.post("/v1/a2a/discover", json={"url": "https://upstream.example"})
    assert resp.status_code == 403


def test_discover_returns_400_when_upstream_unreachable():
    client = _client_for_role(LitellmUserRoles.PROXY_ADMIN)
    with patch(
        "litellm.proxy.a2a.endpoints.fetch_well_known_card",
        new=AsyncMock(side_effect=AgentCardDiscoveryError("no luck")),
    ):
        resp = client.post("/v1/a2a/discover", json={"url": "https://upstream.example"})

    assert resp.status_code == 400
    assert "no luck" in resp.json()["detail"]


def test_discover_forwards_mode_and_params_to_fetcher():
    """The endpoint must hand discovery_mode + params to fetch_well_known_card."""
    client = _client_for_role(LitellmUserRoles.PROXY_ADMIN)
    fetch_stub = AsyncMock(return_value={"name": "support-agent"})
    with patch("litellm.proxy.a2a.endpoints.fetch_well_known_card", new=fetch_stub):
        resp = client.post(
            "/v1/a2a/discover",
            json={
                "url": "http://localhost:2024",
                "discovery_mode": "langgraph_platform",
                "params": {"assistant_id": "agent"},
            },
        )

    assert resp.status_code == 200
    # Pydantic deserializes the JSON string back into the DiscoveryMode enum.
    assert fetch_stub.await_args is not None
    kwargs = fetch_stub.await_args.kwargs
    assert kwargs["discovery_mode"] == DiscoveryMode.LANGGRAPH_PLATFORM
    assert kwargs["params"] == {"assistant_id": "agent"}


def test_discover_rejects_unknown_mode():
    """Pydantic should 422 on an enum value we don't recognize."""
    client = _client_for_role(LitellmUserRoles.PROXY_ADMIN)
    resp = client.post(
        "/v1/a2a/discover",
        json={"url": "http://localhost:2024", "discovery_mode": "bogus"},
    )
    assert resp.status_code == 422
