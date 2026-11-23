import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Final

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.openai.live.handler import OpenAILiveSessions
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth, user_api_key_auth_live_websocket
from litellm.proxy.realtime_endpoints import live_endpoints, live_session_controls
from litellm.proxy.realtime_endpoints.live_endpoints import (
    _LIVE_SESSION_RETENTION_SECONDS,
    _LIVE_SESSION_TOKEN_TTL_SECONDS,
    _LiveSession,
    _session_token,
    _usable_session,
)
from litellm.realtime_api import main as realtime_main

_NOW: Final = 1_800_000_000
_UPSTREAM: Final = "https://api.openai.com/v1/live/sessions"
_OWNER: Final = UserAPIKeyAuth(token="owner-key-hash", team_id="team-a")
_STRANGER: Final = UserAPIKeyAuth(token="other-key-hash", team_id="team-a")
_INBOUND_OPERATOR: Final = UserAPIKeyAuth(token="operator-key-hash", allowed_routes=["/v1/live/sessions/*"])
_ADMIN: Final = UserAPIKeyAuth(token="admin-key-hash", user_role=LitellmUserRoles.PROXY_ADMIN)
_SESSION: Final = _LiveSession(session_id="live_upstream", model="live-group", deployment_id="deployment-1")
_FORK_BODY: Final = {"transport": {"type": "webrtc", "sdp": "v=0 offer"}}
_ACCEPT_BODY: Final = {"session": {"type": "live", "model": "live-group"}}
_TOKEN_CONTROLS: Final = (
    ("POST", "hangup", None),
    ("POST", "refer", {"target_uri": "tel:+15550100"}),
    ("POST", "fork", _FORK_BODY),
    ("GET", "content", None),
)
_EVERY_CONTROL: Final = (
    *_TOKEN_CONTROLS,
    ("POST", "reject", {"status_code": 486}),
    ("POST", "accept", _ACCEPT_BODY),
)
_STARTED: Final = '{"type":"session.started","session":{"id":"live_primary","model":"gpt-live-1"}}'
_RINGING: Final = '{"type":"transport.ringing","session_id":"live_primary","item_id":"live_primary_item"}'
_ClientFor = Callable[[UserAPIKeyAuth], TestClient]


@pytest.fixture(autouse=True)
def _fixed_clock_and_salt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-live-session-test-salt")
    monkeypatch.setattr(live_endpoints.time, "time", lambda: _NOW)


@pytest.fixture
def started_sidebands(monkeypatch: pytest.MonkeyPatch) -> list[_LiveSession]:
    started: Final[list[_LiveSession]] = []  # mutable-ok: collects each billing sideband the route starts

    async def _joins(session: _LiveSession, _auth: UserAPIKeyAuth) -> bool:
        started.append(session)
        return True

    monkeypatch.setattr(live_endpoints, "_start_live_billing_sideband", _joins)
    return started


@pytest.fixture
def client_for(monkeypatch: pytest.MonkeyPatch) -> _ClientFor:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    model_list: Final = [  # mutable-ok: Router takes a list of deployments
        {
            "model_name": "live-group",
            "litellm_params": {"model": "openai/gpt-live-1", "api_key": "sk-upstream"},
            "model_info": {"id": "deployment-1"},
        }
    ]
    monkeypatch.setattr(proxy_server, "llm_router", litellm.Router(model_list=model_list))
    monkeypatch.setattr(proxy_server, "llm_model_list", model_list)

    def _client(caller: UserAPIKeyAuth) -> TestClient:
        async def _authorized(_websocket: object, model: str = "") -> UserAPIKeyAuth:
            return caller

        monkeypatch.setitem(proxy_server.app.dependency_overrides, user_api_key_auth, lambda: caller)
        monkeypatch.setitem(proxy_server.app.dependency_overrides, user_api_key_auth_live_websocket, lambda: caller)
        monkeypatch.setitem(
            proxy_server.app.dependency_overrides, proxy_server.live_session_model_authorizer, lambda: _authorized
        )
        monkeypatch.setattr(proxy_server, "user_api_key_auth_accepted_websocket_for_model", _authorized)
        return TestClient(proxy_server.app, raise_server_exceptions=False)

    return _client


def _call(
    client: TestClient, method: str, session_id: str, action: str, body: object, query: str = ""
) -> httpx.Response:
    url: Final = f"/v1/live/sessions/{session_id}/{action}{query}"
    return client.get(url) if method == "GET" else client.post(url, json=body)


def _close_code(client: TestClient, path: str) -> int:
    with client.websocket_connect(path) as socket:
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_text()
    return closed.value.code


class _RelayUpstream:
    def __init__(self, frames: tuple[str, ...], *, speaks_first: bool = False) -> None:
        self.sent: Final[list[str]] = []  # mutable-ok: records each frame the proxy sends upstream
        self._frames: Final = iter(frames)
        self._start_sent: Final = asyncio.Event()
        if speaks_first:
            self._start_sent.set()

    async def send(self, data: str) -> None:
        self.sent.append(data)
        self._start_sent.set()

    async def recv(self) -> str:
        await self._start_sent.wait()
        frame: Final = next(self._frames, None)
        if frame is None:
            raise ConnectionClosedOK(Close(1000, "done"), Close(1000, "done"), True)
        return frame

    async def close(self) -> None:
        return None

    async def __aenter__(self) -> "_RelayUpstream":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


class _RelayConnector:
    def __init__(self, upstream: _RelayUpstream) -> None:
        self.upstream: Final = upstream
        self.urls: Final[list[str]] = []  # mutable-ok: records each upstream URL the proxy opens

    def __call__(self, url: str, **_kwargs: object) -> _RelayUpstream:
        self.urls.append(url)
        return self.upstream


class _DroppedLoggingWorker:
    def ensure_initialized_and_enqueue(self, async_coroutine: object) -> None:
        close: Final = getattr(async_coroutine, "close", None)
        if callable(close):
            close()


def _relay_through(
    monkeypatch: pytest.MonkeyPatch, frames: tuple[str, ...], *, speaks_first: bool = False
) -> _RelayConnector:
    connector: Final = _RelayConnector(_RelayUpstream(frames, speaks_first=speaks_first))
    monkeypatch.setattr(
        realtime_main,
        "openai_live_sessions",
        OpenAILiveSessions(logging_worker=_DroppedLoggingWorker(), websocket_connector=connector),
    )
    return connector


@pytest.mark.parametrize(("method", "action", "body"), _EVERY_CONTROL, ids=[control[1] for control in _EVERY_CONTROL])
@pytest.mark.parametrize("token_kind", ["foreign", "expired", "garbage"])
def test_a_bad_token_gets_the_same_404_on_every_control_and_never_reaches_the_provider(
    client_for: _ClientFor,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    action: str,
    body: object,
    token_kind: str,
) -> None:
    token: Final = {
        "foreign": _session_token(_SESSION, _STRANGER),
        "expired": _session_token(_SESSION, _OWNER),
        "garbage": "not-a-session-token",
    }[token_kind]
    if token_kind == "expired":
        monkeypatch.setattr(live_endpoints.time, "time", lambda: _NOW + _LIVE_SESSION_TOKEN_TTL_SECONDS + 1)

    with respx.mock(assert_all_called=False) as provider:
        upstream: Final = provider.route(host="api.openai.com").mock(return_value=httpx.Response(200))
        response: Final = _call(client_for(_OWNER), method, token, action, body)

    assert response.status_code == 404, response.text
    assert response.json() == {"detail": "Session not found"}
    assert not upstream.called


@pytest.mark.parametrize(("method", "action", "body"), _TOKEN_CONTROLS, ids=[control[1] for control in _TOKEN_CONTROLS])
def test_a_token_whose_deployment_is_gone_is_a_404(
    client_for: _ClientFor, method: str, action: str, body: object
) -> None:
    token: Final = _session_token(
        _LiveSession(session_id="live_upstream", model="live-group", deployment_id="deployment-removed"), _OWNER
    )

    with respx.mock(assert_all_called=False) as provider:
        upstream: Final = provider.route(host="api.openai.com").mock(return_value=httpx.Response(200))
        response: Final = _call(client_for(_OWNER), method, token, action, body)

    assert response.status_code == 404, response.text
    assert not upstream.called


@pytest.mark.parametrize(
    ("method", "action", "body"),
    [control for control in _TOKEN_CONTROLS if control[1] != "hangup"],
    ids=[control[1] for control in _TOKEN_CONTROLS if control[1] != "hangup"],
)
def test_an_owner_who_lost_the_model_cannot_control_the_session(
    client_for: _ClientFor, method: str, action: str, body: object
) -> None:
    owner_without_model: Final = UserAPIKeyAuth(token="owner-key-hash", team_id="team-a", models=["another-model"])
    token: Final = _session_token(_SESSION, owner_without_model)

    with respx.mock(assert_all_called=False) as provider:
        upstream: Final = provider.route(host="api.openai.com").mock(return_value=httpx.Response(200))
        response: Final = _call(client_for(owner_without_model), method, token, action, body)

    assert response.status_code >= 400, response.text
    assert not upstream.called


def test_an_owner_who_lost_the_model_can_still_end_a_running_call(
    client_for: _ClientFor, respx_mock: respx.MockRouter
) -> None:
    owner_without_model: Final = UserAPIKeyAuth(token="owner-key-hash", team_id="team-a", models=["another-model"])
    hangup: Final = respx_mock.post(f"{_UPSTREAM}/live_upstream/hangup").mock(return_value=httpx.Response(200))

    response: Final = _call(
        client_for(owner_without_model), "POST", _session_token(_SESSION, owner_without_model), "hangup", None
    )

    assert response.status_code == 200, response.text
    assert hangup.call_count == 1


@pytest.mark.parametrize(
    "allowed_routes",
    [None, [], ["/v1/*"], ["/*"], ["llm_api_routes"], ["/v1/live/sessions"]],
    ids=["none", "empty", "v1-wildcard", "all-wildcard", "route-group", "create-only"],
)
@pytest.mark.parametrize(("method", "action", "body"), _EVERY_CONTROL, ids=[control[1] for control in _EVERY_CONTROL])
def test_a_key_without_the_inbound_grant_cannot_use_a_raw_id(
    client_for: _ClientFor, method: str, action: str, body: object, allowed_routes: list[str] | None
) -> None:
    ordinary_key: Final = UserAPIKeyAuth(token="ordinary-key-hash", allowed_routes=allowed_routes)

    with respx.mock(assert_all_called=False) as provider:
        upstream: Final = provider.route(host="api.openai.com").mock(return_value=httpx.Response(200))
        response: Final = _call(client_for(ordinary_key), method, "live_inbound", action, body, "?model=live-group")

    assert response.status_code == 404, response.text
    assert not upstream.called


@pytest.mark.parametrize("allowed_routes", [None, [], ["/v1/*"]], ids=["none", "empty", "v1-wildcard"])
def test_a_key_without_the_inbound_grant_cannot_attach_to_a_raw_id(
    client_for: _ClientFor, monkeypatch: pytest.MonkeyPatch, allowed_routes: list[str] | None
) -> None:
    connector: Final = _relay_through(monkeypatch, ())
    ordinary_key: Final = UserAPIKeyAuth(token="ordinary-key-hash", allowed_routes=allowed_routes)

    code: Final = _close_code(client_for(ordinary_key), "/v1/live/sessions/live_inbound/attach?model=live-group")

    assert code == 1008
    assert connector.urls == []


@pytest.mark.parametrize("caller", [_INBOUND_OPERATOR, _ADMIN], ids=["granted-key", "proxy-admin"])
def test_a_granted_caller_attaches_to_a_raw_id_with_the_model_query(
    client_for: _ClientFor, monkeypatch: pytest.MonkeyPatch, caller: UserAPIKeyAuth
) -> None:
    connector: Final = _relay_through(monkeypatch, (_STARTED,), speaks_first=True)

    with client_for(caller).websocket_connect("/v1/live/sessions/live_inbound/attach?model=live-group") as socket:
        received: Final = socket.receive_text()

    assert received == _STARTED
    assert connector.urls == ["wss://api.openai.com/v1/live/sessions/live_inbound/attach"]
    assert connector.upstream.sent == [], "an attached socket must not start the caller's session"


def test_a_granted_caller_attaching_to_a_raw_id_without_the_model_query_is_refused(
    client_for: _ClientFor, monkeypatch: pytest.MonkeyPatch
) -> None:
    connector: Final = _relay_through(monkeypatch, ())

    code: Final = _close_code(client_for(_INBOUND_OPERATOR), "/v1/live/sessions/live_inbound/attach")

    assert code == 1008
    assert connector.urls == []


@pytest.mark.parametrize("prefix", ["/live/sessions", "/v1/live/sessions"])
def test_a_control_works_on_every_route_prefix(
    client_for: _ClientFor, respx_mock: respx.MockRouter, prefix: str
) -> None:
    hangup: Final = respx_mock.post(f"{_UPSTREAM}/live_upstream/hangup").mock(return_value=httpx.Response(200))

    response: Final = client_for(_OWNER).post(f"{prefix}/{_session_token(_SESSION, _OWNER)}/hangup")

    assert response.status_code == 200, response.text
    assert hangup.call_count == 1


@pytest.mark.parametrize("caller", [_INBOUND_OPERATOR, _ADMIN], ids=["granted-key", "proxy-admin"])
def test_a_granted_caller_accepts_a_raw_id_and_the_call_is_billed_to_it(
    client_for: _ClientFor,
    started_sidebands: list[_LiveSession],
    respx_mock: respx.MockRouter,
    caller: UserAPIKeyAuth,
) -> None:
    accepted: Final = respx_mock.post(f"{_UPSTREAM}/live_inbound/accept").mock(return_value=httpx.Response(200))

    response: Final = client_for(caller).post("/v1/live/sessions/live_inbound/accept", json=_ACCEPT_BODY)

    assert response.status_code == 200, response.text
    assert response.content == b""
    assert accepted.call_count == 1
    assert json.loads(accepted.calls.last.request.content) == {"session": {"type": "live", "model": "gpt-live-1"}}
    assert started_sidebands == [
        _LiveSession(session_id="live_inbound", model="live-group", deployment_id="deployment-1")
    ]


def test_an_accepted_call_whose_billing_cannot_join_is_hung_up_once(
    client_for: _ClientFor, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _does_not_join(_session: _LiveSession, _auth: UserAPIKeyAuth) -> bool:
        return False

    monkeypatch.setattr(live_endpoints, "_start_live_billing_sideband", _does_not_join)
    respx_mock.post(f"{_UPSTREAM}/live_inbound/accept").mock(return_value=httpx.Response(200))
    hangup: Final = respx_mock.post(f"{_UPSTREAM}/live_inbound/hangup").mock(return_value=httpx.Response(200))

    response: Final = client_for(_INBOUND_OPERATOR).post("/v1/live/sessions/live_inbound/accept", json=_ACCEPT_BODY)

    assert response.status_code == 503, response.text
    assert hangup.call_count == 1


@pytest.mark.parametrize("action", ["reject", "hangup", "refer"])
def test_a_raw_id_without_the_model_query_is_a_400_that_says_so(client_for: _ClientFor, action: str) -> None:
    with respx.mock(assert_all_called=False) as provider:
        upstream: Final = provider.route(host="api.openai.com").mock(return_value=httpx.Response(200))
        response: Final = client_for(_INBOUND_OPERATOR).post(f"/v1/live/sessions/live_inbound/{action}", json={})

    assert response.status_code == 400, response.text
    assert "model query parameter" in response.text
    assert not upstream.called


def test_a_granted_caller_rejects_a_raw_id_with_the_model_query(
    client_for: _ClientFor, respx_mock: respx.MockRouter
) -> None:
    rejected: Final = respx_mock.post(f"{_UPSTREAM}/live_inbound/reject").mock(return_value=httpx.Response(200))

    response: Final = client_for(_INBOUND_OPERATOR).post(
        "/v1/live/sessions/live_inbound/reject?model=live-group", json={"status_code": 486}
    )

    assert response.status_code == 200, response.text
    assert json.loads(rejected.calls.last.request.content) == {"status_code": 486}
    assert rejected.calls.last.request.headers["authorization"] == "Bearer sk-upstream"


def test_hangup_with_a_token_reaches_the_pinned_deployment_with_no_body(
    client_for: _ClientFor, respx_mock: respx.MockRouter
) -> None:
    hangup: Final = respx_mock.post(f"{_UPSTREAM}/live_upstream/hangup").mock(return_value=httpx.Response(200))

    response: Final = client_for(_OWNER).post(f"/v1/live/sessions/{_session_token(_SESSION, _OWNER)}/hangup")

    assert response.status_code == 200, response.text
    assert response.content == b""
    assert hangup.calls.last.request.content == b""
    assert hangup.calls.last.request.headers["authorization"] == "Bearer sk-upstream"


def test_refer_hands_back_the_providers_error(client_for: _ClientFor, respx_mock: respx.MockRouter) -> None:
    respx_mock.post(f"{_UPSTREAM}/live_upstream/refer").mock(
        return_value=httpx.Response(409, json={"error": {"message": "call is not active"}})
    )

    response: Final = client_for(_OWNER).post(
        f"/v1/live/sessions/{_session_token(_SESSION, _OWNER)}/refer", json={"target_uri": "tel:+15550100"}
    )

    assert response.status_code == 409, response.text
    assert response.json() == {"error": {"message": "call is not active"}}


def test_a_stored_session_keeps_fork_and_content_after_attach_and_live_controls_expire(
    client_for: _ClientFor,
    started_sidebands: list[_LiveSession],
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token: Final = _session_token(_SESSION, _OWNER, stored=True)
    respx_mock.get(f"{_UPSTREAM}/live_upstream/content").mock(
        return_value=httpx.Response(200, content=b"RIFF-wav", headers={"content-type": "audio/wav"})
    )
    respx_mock.post(f"{_UPSTREAM}/live_upstream/fork").mock(
        return_value=httpx.Response(
            201, json={"session": {"id": "live_forked"}, "transport": {"type": "webrtc", "sdp": "v=0 answer"}}
        )
    )
    monkeypatch.setattr(live_endpoints.time, "time", lambda: _NOW + _LIVE_SESSION_TOKEN_TTL_SECONDS + 1)
    client: Final = client_for(_OWNER)

    content: Final = client.get(f"/v1/live/sessions/{token}/content")
    fork: Final = client.post(f"/v1/live/sessions/{token}/fork", json=_FORK_BODY)
    hangup: Final = client.post(f"/v1/live/sessions/{token}/hangup")

    assert (content.status_code, content.content) == (200, b"RIFF-wav")
    assert content.headers["content-type"].startswith("audio/wav")
    assert fork.status_code == 201, fork.text
    assert hangup.status_code == 404, hangup.text
    assert _close_code(client, f"/v1/live/sessions/{token}/attach") == 1008


def test_a_stored_session_is_gone_after_its_retention(
    client_for: _ClientFor, monkeypatch: pytest.MonkeyPatch
) -> None:
    token: Final = _session_token(_SESSION, _OWNER, stored=True)
    monkeypatch.setattr(live_endpoints.time, "time", lambda: _NOW + _LIVE_SESSION_RETENTION_SECONDS + 1)

    with respx.mock(assert_all_called=False) as provider:
        upstream: Final = provider.route(host="api.openai.com").mock(return_value=httpx.Response(200))
        response: Final = client_for(_OWNER).get(f"/v1/live/sessions/{token}/content")

    assert response.status_code == 404, response.text
    assert not upstream.called


def test_content_hands_back_the_providers_error(client_for: _ClientFor, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{_UPSTREAM}/live_upstream/content").mock(
        return_value=httpx.Response(404, json={"error": {"message": "no recording"}})
    )

    token: Final = _session_token(_SESSION, _OWNER, stored=True)

    response: Final = client_for(_OWNER).get(f"/v1/live/sessions/{token}/content")

    assert response.status_code == 404, response.text
    assert response.json() == {"error": {"message": "no recording"}}


class _TrackedRecording(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.chunks_read = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in (b"first", b"second", b"third"):
            self.chunks_read += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_a_recording_download_stops_reading_and_closes_upstream_when_the_client_goes_away() -> None:
    recording: Final = _TrackedRecording()
    upstream: Final = httpx.Response(200, headers={"content-type": "audio/wav"}, stream=recording)

    async def _receive() -> dict[str, object]:
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def _client_gone_on_first_chunk(message: dict[str, object]) -> None:
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    with pytest.raises(ClientDisconnect):
        await live_session_controls._UpstreamRecording(upstream)(
            {"type": "http", "asgi": {"spec_version": "2.4"}}, _receive, _client_gone_on_first_chunk
        )

    assert recording.chunks_read == 1
    assert recording.closed


def test_a_fork_gets_a_new_token_is_billed_and_sends_no_model_to_the_provider(
    client_for: _ClientFor, started_sidebands: list[_LiveSession], respx_mock: respx.MockRouter
) -> None:
    fork: Final = respx_mock.post(f"{_UPSTREAM}/live_upstream/fork").mock(
        return_value=httpx.Response(
            201, json={"session": {"id": "live_forked"}, "transport": {"type": "webrtc", "sdp": "v=0 answer"}}
        )
    )
    token: Final = _session_token(_SESSION, _OWNER, stored=True)

    response: Final = client_for(_OWNER).post(
        f"/v1/live/sessions/{token}/fork",
        json={**_FORK_BODY, "session": {"model": "live-group", "instructions": "Continue"}},
    )

    forked: Final = _LiveSession(session_id="live_forked", model="live-group", deployment_id="deployment-1")
    assert response.status_code == 201, response.text
    assert json.loads(fork.calls.last.request.content)["session"] == {"instructions": "Continue"}
    opened: Final = _usable_session(response.json()["session"]["id"], _OWNER, retained=True)
    assert opened is not None
    assert opened.session == forked
    assert opened.retained_until is not None, "a fork of a stored session inherits storage when it does not override it"
    assert started_sidebands == [forked]


def test_a_fork_cannot_move_to_another_model_group(
    client_for: _ClientFor, started_sidebands: list[_LiveSession]
) -> None:
    token: Final = _session_token(_SESSION, _OWNER, stored=True)

    with respx.mock(assert_all_called=False) as provider:
        upstream: Final = provider.route(host="api.openai.com").mock(return_value=httpx.Response(201))
        response: Final = client_for(_OWNER).post(
            f"/v1/live/sessions/{token}/fork", json={**_FORK_BODY, "session": {"model": "other-group"}}
        )

    assert response.status_code == 400, response.text
    assert not upstream.called
    assert started_sidebands == []


def test_a_websocket_session_without_store_keeps_every_frame_byte_identical(
    client_for: _ClientFor, monkeypatch: pytest.MonkeyPatch
) -> None:
    _relay_through(monkeypatch, (_STARTED, _RINGING))

    with client_for(_OWNER).websocket_connect("/v1/live/sessions") as socket:
        socket.send_json({"type": "session.start", "session": {"model": "live-group"}})
        received: Final = (socket.receive_text(), socket.receive_text())

    assert received == (_STARTED, _RINGING)


def test_a_stored_websocket_session_shows_a_token_for_its_id_in_every_frame(
    client_for: _ClientFor, monkeypatch: pytest.MonkeyPatch
) -> None:
    _relay_through(monkeypatch, (_STARTED, _RINGING))

    with client_for(_OWNER).websocket_connect("/v1/live/sessions") as socket:
        socket.send_json({"type": "session.start", "session": {"model": "live-group", "store": True}})
        started: Final = socket.receive_json()
        ringing: Final = socket.receive_json()

    token: Final = started["session"]["id"]
    opened: Final = _usable_session(token, _OWNER, retained=True)
    assert opened is not None
    assert opened.session == _LiveSession(session_id="live_primary", model="live-group", deployment_id="deployment-1")
    assert opened.retained_until is not None
    assert ringing["session_id"] == token
    assert ringing["item_id"] == "live_primary_item", "a different id that contains this one must stay as it is"


@pytest.mark.parametrize(
    ("store_override", "keeps_token"),
    [(None, True), (True, True), (False, False)],
    ids=["inherits-store", "store-on", "store-off"],
)
def test_a_socket_fork_relays_to_the_source_and_keeps_storage_as_asked(
    client_for: _ClientFor,
    monkeypatch: pytest.MonkeyPatch,
    store_override: bool | None,
    keeps_token: bool,
) -> None:
    connector: Final = _relay_through(monkeypatch, (_STARTED,))
    token: Final = _session_token(_SESSION, _OWNER, stored=True)
    overrides: Final = {"instructions": "Continue", **({} if store_override is None else {"store": store_override})}

    with client_for(_OWNER).websocket_connect(f"/v1/live/sessions/{token}/fork") as socket:
        socket.send_json({"type": "session.start", "session": overrides})
        started: Final = socket.receive_json()

    assert connector.urls == ["wss://api.openai.com/v1/live/sessions/live_upstream/fork"]
    assert json.loads(connector.upstream.sent[0]) == {"type": "session.start", "session": overrides}
    assert (started["session"]["id"] != "live_primary") is keeps_token


@pytest.mark.parametrize(
    "first_frame",
    ['{"type":"session.start","session":{"model":"other-group"}}', '{"type":"other"}', "not json"],
    ids=["other-model-group", "wrong-type", "not-json"],
)
def test_a_socket_fork_refuses_a_bad_first_frame(
    client_for: _ClientFor, monkeypatch: pytest.MonkeyPatch, first_frame: str
) -> None:
    connector: Final = _relay_through(monkeypatch, ())
    token: Final = _session_token(_SESSION, _OWNER, stored=True)

    with client_for(_OWNER).websocket_connect(f"/v1/live/sessions/{token}/fork") as socket:
        socket.send_text(first_frame)
        error: Final = socket.receive_json()
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_text()

    assert error["error"]["code"] == "invalid_session_start"
    assert closed.value.code == 1008
    assert connector.urls == []


def test_a_socket_fork_with_a_foreign_token_never_opens_upstream(
    client_for: _ClientFor, monkeypatch: pytest.MonkeyPatch
) -> None:
    connector: Final = _relay_through(monkeypatch, ())
    token: Final = _session_token(_SESSION, _STRANGER, stored=True)

    assert _close_code(client_for(_OWNER), f"/v1/live/sessions/{token}/fork") == 1008
    assert connector.urls == []
