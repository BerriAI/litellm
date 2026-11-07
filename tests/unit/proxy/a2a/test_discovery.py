"""Tests for the well-known card fetcher and the discovery endpoint."""

from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.a2a.discovery import (
    AGENT_CARD_WELL_KNOWN_PATHS,
    AgentCardDiscoveryError,
    DiscoveryMode,
    card_request_headers,
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


# ---------------------------------------------------------------------------
# Discovery authenticates like the saved agent
# ---------------------------------------------------------------------------

_UPSTREAM: Final = "https://upstream.example"
_CARD: Final = {"name": "upstream-agent"}
_BASIC: Final = {"authorization": "Basic dXNlcjpwdw=="}


@pytest.mark.parametrize(
    ("api_key", "static_headers", "mode", "expected"),
    [
        (
            "sk-typed",
            {"x-tenant": "t1"},
            DiscoveryMode.WELL_KNOWN_FALLBACK,
            {"Authorization": "Bearer sk-typed", "x-tenant": "t1"},
        ),
        ("sk-typed", _BASIC, DiscoveryMode.WELL_KNOWN_FALLBACK, _BASIC),
        (
            "sk-typed",
            {**_BASIC, "x-tenant": "t1"},
            DiscoveryMode.LANGGRAPH_PLATFORM,
            {"x-tenant": "t1", "Authorization": "Bearer sk-typed"},
        ),
        (None, _BASIC, DiscoveryMode.LANGGRAPH_PLATFORM, _BASIC),
        ("", {}, DiscoveryMode.WELL_KNOWN_FALLBACK, {}),
    ],
)
def test_card_request_headers_authenticate_like_the_agents_runtime(
    api_key: str | None, static_headers: Mapping[str, str], mode: DiscoveryMode, expected: Mapping[str, str]
) -> None:
    assert dict(card_request_headers(api_key, static_headers, mode)) == expected


def _card_behind_auth(request: httpx.Request) -> httpx.Response:
    if "authorization" in request.headers:
        return httpx.Response(200, json=_CARD)
    return httpx.Response(401, json={"detail": "Unauthorized"})


@pytest.mark.usefixtures("httpx_transport")
@respx.mock
def test_discover_sends_the_key_and_static_headers_upstream() -> None:
    route: Final = respx.get(f"{_UPSTREAM}/.well-known/agent-card.json").mock(side_effect=_card_behind_auth)

    response: Final = _client_for_role(LitellmUserRoles.PROXY_ADMIN).post(
        "/v1/a2a/discover",
        json={"url": _UPSTREAM, "api_key": "sk-typed", "static_headers": {"x-tenant": "t1"}},
    )

    assert (response.status_code, response.json()["agent_card"]) == (200, _CARD)
    sent: Final = [(call.request.headers.get("authorization"), call.request.headers.get("x-tenant")) for call in route.calls]
    assert sent == [(None, None), ("Bearer sk-typed", "t1")]


def _public_card_unless_authorized(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_CARD if "authorization" in request.headers else {"name": "public-agent"})


@pytest.mark.asyncio
@pytest.mark.usefixtures("httpx_transport")
@respx.mock
async def test_fetch_sends_its_headers_with_every_request_as_before() -> None:
    route: Final = respx.get(f"{_UPSTREAM}/.well-known/agent-card.json").mock(side_effect=_public_card_unless_authorized)

    card: Final = await fetch_well_known_card(_UPSTREAM, headers={"Authorization": "Bearer sk-1"})

    assert card == _CARD
    assert [call.request.headers.get("authorization") for call in route.calls] == ["Bearer sk-1"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("httpx_transport")
@respx.mock
async def test_fetch_keeps_its_headers_on_the_credentialed_retry() -> None:
    route: Final = respx.get(f"{_UPSTREAM}/.well-known/agent-card.json").mock(side_effect=_card_behind_auth)

    card: Final = await fetch_well_known_card(
        _UPSTREAM, headers={"x-tenant": "t1"}, credentials={"Authorization": "Bearer sk-1"}
    )

    assert card == _CARD
    sent: Final = [(call.request.headers.get("authorization"), call.request.headers.get("x-tenant")) for call in route.calls]
    assert sent == [(None, "t1"), ("Bearer sk-1", "t1")]


@pytest.mark.usefixtures("httpx_transport")
@respx.mock
def test_discover_returns_the_card_the_credentials_unlock_before_a_later_public_one() -> None:
    """The saved agent's own client also asks this path first, with its credentials."""
    respx.get(f"{_UPSTREAM}/.well-known/agent-card.json").mock(side_effect=_card_behind_auth)
    respx.get(f"{_UPSTREAM}/.well-known/agent.json").respond(200, json={"name": "public-agent"})

    response: Final = _client_for_role(LitellmUserRoles.PROXY_ADMIN).post(
        "/v1/a2a/discover", json={"url": _UPSTREAM, "api_key": "sk-typed"}
    )

    assert (response.status_code, response.json()["agent_card"]) == (200, _CARD)


_LOCAL_AGENT: Final = "http://127.0.0.1:8123"
_OTHER_ORIGIN: Final = "http://127.0.0.1:8124"


@pytest.mark.parametrize("validate_urls", [True, False])
@pytest.mark.usefixtures("httpx_transport")
def test_discover_never_sends_credentials_to_a_redirect_target(
    monkeypatch: pytest.MonkeyPatch, validate_urls: bool
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", validate_urls)
    monkeypatch.setattr(litellm, "user_url_allowed_hosts", ["127.0.0.1:8123", "127.0.0.1:8124"])

    def agent(request: httpx.Request) -> httpx.Response:
        if "x-api-key" in request.headers:
            return httpx.Response(302, headers={"Location": f"{_OTHER_ORIGIN}/card"})
        return httpx.Response(401)

    with respx.mock(assert_all_called=False) as router:
        router.get(url__startswith=_LOCAL_AGENT).mock(side_effect=agent)
        elsewhere: Final = router.get(url__startswith=_OTHER_ORIGIN).respond(200, json=_CARD)
        response: Final = _client_for_role(LitellmUserRoles.PROXY_ADMIN).post(
            "/v1/a2a/discover", json={"url": _LOCAL_AGENT, "static_headers": {"x-api-key": "secret"}}
        )

    assert response.status_code == 400
    assert elsewhere.call_count == 0


@pytest.mark.parametrize("status", [200, 203])
@pytest.mark.usefixtures("httpx_transport")
@respx.mock
def test_discover_loads_a_public_card_exactly_as_before_without_sending_credentials(status: int) -> None:
    route: Final = respx.get(f"{_UPSTREAM}/.well-known/agent-card.json").respond(status, json=_CARD)

    response: Final = _client_for_role(LitellmUserRoles.PROXY_ADMIN).post(
        "/v1/a2a/discover", json={"url": _UPSTREAM, "api_key": "sk-1", "static_headers": {"x-tenant": "t1"}}
    )

    assert (response.status_code, response.json()["agent_card"]) == (200, _CARD)
    assert [dict(call.request.headers).keys() & {"authorization", "x-tenant"} for call in route.calls] == [set()]


@pytest.mark.parametrize("anonymous_body", ["<html>sign in</html>", "[" * 100_000 + "]" * 100_000])
@pytest.mark.usefixtures("httpx_transport")
@respx.mock
def test_discover_sends_credentials_when_the_anonymous_answer_is_not_a_card(anonymous_body: str) -> None:
    respx.get(f"{_UPSTREAM}/.well-known/agent-card.json").mock(
        side_effect=lambda request: (
            httpx.Response(200, json=_CARD)
            if "authorization" in request.headers
            else httpx.Response(200, text=anonymous_body)
        )
    )

    response: Final = _client_for_role(LitellmUserRoles.PROXY_ADMIN).post(
        "/v1/a2a/discover", json={"url": _UPSTREAM, "api_key": "sk-1"}
    )

    assert (response.status_code, response.json()["agent_card"]) == (200, _CARD)


@pytest.mark.parametrize(
    "rejection",
    [
        httpx.Response(401),
        httpx.Response(200, text="<html>sign in</html>"),
        httpx.Response(200, json=["not", "a card"]),
        httpx.Response(302, headers={"Location": f"{_UPSTREAM}/elsewhere"}, json={"moved": "elsewhere"}),
        httpx.LocalProtocolError("Illegal header value"),
        ValueError("Newline in header value"),
        httpx.ReadTimeout("timed out"),
    ],
)
@pytest.mark.usefixtures("httpx_transport")
@respx.mock
def test_discover_reports_the_anonymous_answer_when_the_credentials_fail(rejection: httpx.Response | Exception) -> None:
    def card(request: httpx.Request) -> httpx.Response:
        if "authorization" not in request.headers:
            return httpx.Response(403)
        if isinstance(rejection, Exception):
            raise rejection
        return rejection

    respx.route().mock(side_effect=card)

    response: Final = _client_for_role(LitellmUserRoles.PROXY_ADMIN).post(
        "/v1/a2a/discover", json={"url": _UPSTREAM, "api_key": "sk-1"}
    )

    assert (response.status_code, response.json()["detail"]) == (
        400,
        "Could not fetch agent card from https://upstream.example (mode=well_known_fallback). "
        "Last error: https://upstream.example/agent.json: HTTP 403",
    )


@pytest.mark.usefixtures("httpx_transport")
@respx.mock
def test_discover_does_not_wait_twice_for_an_unresponsive_agent() -> None:
    timing_out: Final = respx.route().mock(side_effect=httpx.ReadTimeout("timed out"))

    response: Final = _client_for_role(LitellmUserRoles.PROXY_ADMIN).post(
        "/v1/a2a/discover", json={"url": _UPSTREAM, "api_key": "sk-1"}
    )

    assert response.status_code == 400
    assert timing_out.call_count == len(AGENT_CARD_WELL_KNOWN_PATHS)
