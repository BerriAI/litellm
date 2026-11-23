import json
import logging
from typing import Final

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.realtime_endpoints import live_endpoints
from litellm.proxy.realtime_endpoints.live_endpoints import _attachable_session, _LiveSession, _session_token

_NOW: Final = 1_800_000_000
_OWNER: Final = UserAPIKeyAuth(token="owner-key-hash", team_id="team-a")
_SESSION: Final = _LiveSession(session_id="live_upstream", model="gpt-live-1", deployment_id="deployment-1")


@pytest.fixture(autouse=True)
def _fixed_clock_and_salt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-live-session-test-salt")
    monkeypatch.setattr(live_endpoints.time, "time", lambda: _NOW)


def test_session_token_hides_the_upstream_id_and_opens_for_its_owner() -> None:
    token: Final = _session_token(_SESSION, _OWNER)

    assert "live_upstream" not in token
    assert _attachable_session(token, _OWNER) == _SESSION


@pytest.mark.parametrize(
    ("caller", "expected"),
    [
        (UserAPIKeyAuth(token="owner-key-hash", team_id="team-a"), _SESSION),
        (UserAPIKeyAuth(token="other-key-hash", team_id="team-a"), None),
        (UserAPIKeyAuth(token="other-key-hash", team_id="team-b"), None),
        (UserAPIKeyAuth(token="other-key-hash"), None),
        (UserAPIKeyAuth(), None),
    ],
    ids=["same-key", "teammate-key", "other-team", "other-key-no-team", "no-identity"],
)
def test_session_token_opens_only_for_the_key_that_created_the_session(
    caller: UserAPIKeyAuth, expected: _LiveSession | None
) -> None:
    assert _attachable_session(_session_token(_SESSION, _OWNER), caller) == expected


def test_session_token_from_a_key_without_a_team_does_not_open_for_another_teamless_key() -> None:
    token: Final = _session_token(_SESSION, UserAPIKeyAuth(token="owner-key-hash"))

    assert _attachable_session(token, UserAPIKeyAuth(token="other-key-hash")) is None


def test_expired_session_token_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    token: Final = _session_token(_SESSION, _OWNER)
    monkeypatch.setattr(live_endpoints.time, "time", lambda: _NOW + live_endpoints._LIVE_SESSION_TOKEN_TTL_SECONDS + 1)

    assert _attachable_session(token, _OWNER) is None


def _token_with(**overrides: object) -> str:
    decrypted: Final = decrypt_value_helper(value=_session_token(_SESSION, _OWNER), key="live_session_id")
    assert decrypted is not None
    return encrypt_value_helper(json.dumps({**json.loads(decrypted), **overrides}))


@pytest.mark.parametrize("token", ["live_upstream", ""], ids=["raw-upstream-id", "empty"])
def test_a_session_id_the_proxy_did_not_issue_is_rejected(token: str) -> None:
    assert _attachable_session(token, _OWNER) is None


@pytest.mark.parametrize("plaintext", ["not json", "[]"], ids=["not-json", "not-an-object"])
def test_an_encrypted_value_that_is_not_a_session_token_is_rejected(plaintext: str) -> None:
    assert _attachable_session(encrypt_value_helper(plaintext), _OWNER) is None


@pytest.mark.parametrize(
    "overrides",
    [{"v": "v1"}, {"session_id": 7}, {"model": None}, {"deployment_id": None}, {"expires_at": "soon"}],
    ids=["realtime-token-version", "bad-session-id", "no-model", "no-deployment", "bad-expiry"],
)
def test_a_malformed_session_token_is_rejected(overrides: dict[str, object]) -> None:
    assert _attachable_session(_token_with(**overrides), _OWNER) is None


_CREATE_BODY: Final = {"session": {"model": "live-group"}, "transport": {"type": "webrtc", "sdp": "v=0 offer"}}


@pytest.fixture
def started_sidebands(monkeypatch: pytest.MonkeyPatch) -> list[_LiveSession]:
    started: Final[list[_LiveSession]] = []  # mutable-ok: collects each billing sideband the route starts

    async def _joins(session: _LiveSession, _auth: UserAPIKeyAuth) -> bool:
        started.append(session)
        return True

    monkeypatch.setattr(live_endpoints, "_start_live_billing_sideband", _joins)
    return started


@pytest.fixture
def live_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
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
    monkeypatch.setitem(proxy_server.app.dependency_overrides, user_api_key_auth, lambda: _OWNER)
    return TestClient(proxy_server.app, raise_server_exceptions=False)


def test_create_answers_with_the_sdp_and_a_token_for_the_deployment_that_serves_the_session(
    live_client: TestClient, started_sidebands: list[_LiveSession], respx_mock: respx.MockRouter
) -> None:
    upstream: Final = respx_mock.post("https://api.openai.com/v1/live/sessions").mock(
        return_value=httpx.Response(
            201,
            json={
                "session": {"id": "live_upstream", "model": "gpt-live-1"},
                "transport": {"type": "webrtc", "sdp": "v=0 answer"},
            },
        )
    )

    response: Final = live_client.post("/v1/live/sessions", json=_CREATE_BODY)

    expected_session: Final = _LiveSession(session_id="live_upstream", model="live-group", deployment_id="deployment-1")
    assert response.status_code == 201, response.text
    created: Final = response.json()
    assert created["transport"] == {"type": "webrtc", "sdp": "v=0 answer"}
    assert created["session"]["model"] == "live-group"
    assert _attachable_session(created["session"]["id"], _OWNER) == expected_session
    assert started_sidebands == [expected_session]
    assert upstream.calls.last.request.headers["authorization"] == "Bearer sk-upstream"
    assert json.loads(upstream.calls.last.request.content) == {
        "session": {"model": "gpt-live-1"},
        "transport": {"type": "webrtc", "sdp": "v=0 offer"},
    }


def test_create_hands_back_the_providers_error_and_meters_nothing(
    live_client: TestClient, started_sidebands: list[_LiveSession], respx_mock: respx.MockRouter
) -> None:
    respx_mock.post("https://api.openai.com/v1/live/sessions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}})
    )

    response: Final = live_client.post("/v1/live/sessions", json=_CREATE_BODY)

    assert response.status_code == 401, response.text
    assert response.json() == {"error": {"message": "Incorrect API key provided"}}
    assert started_sidebands == []


def test_create_refuses_a_session_a_guardrail_would_have_to_check(
    live_client: TestClient,
    started_sidebands: list[_LiveSession],
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream: Final = respx_mock.post("https://api.openai.com/v1/live/sessions").mock(
        return_value=httpx.Response(201, json={"session": {"id": "live_upstream"}})
    )
    monkeypatch.setattr(
        litellm, "callbacks", [CustomGuardrail(guardrail_name="blocker", event_hook="pre_call", default_on=True)]
    )

    response: Final = live_client.post("/v1/live/sessions", json=_CREATE_BODY)

    assert response.status_code == 400, response.text
    assert "Guardrails are not supported on OpenAI Live sessions" in response.text
    assert upstream.call_count == 0, "the provider session must not exist when the proxy cannot guard or bill it"
    assert started_sidebands == []


@pytest.mark.parametrize("body", [{}, {"session": {}}, {"session": {"model": ""}}], ids=["empty", "no-model", "blank"])
def test_create_requires_the_session_model(
    live_client: TestClient, started_sidebands: list[_LiveSession], body: dict[str, object]
) -> None:
    response: Final = live_client.post("/v1/live/sessions", json=body)

    assert response.status_code == 400, response.text
    assert "session.model is required" in response.text
    assert started_sidebands == []


@pytest.mark.parametrize("transport", [None, "sip", 7], ids=["missing", "string", "number"])
def test_create_refuses_a_transport_that_is_not_an_object_without_calling_the_provider(
    live_client: TestClient, started_sidebands: list[_LiveSession], transport: object
) -> None:
    body: Final = {"session": {"model": "live-group"}, **({} if transport is None else {"transport": transport})}
    with respx.mock(assert_all_called=False) as provider:
        create: Final = provider.post("https://api.openai.com/v1/live/sessions").mock(return_value=httpx.Response(201))
        response: Final = live_client.post("/v1/live/sessions", json=body)

    assert response.status_code == 400, response.text
    assert "transport must be an object" in response.text
    assert not create.called
    assert started_sidebands == []


_TRUNK_PASSWORD: Final = "trunk-password-do-not-leak"
_SIP_TRANSPORT: Final = {
    "type": "sip",
    "destination": "tel:+15550100",
    "trunk": {"uri": "sip:trunk.example.com", "auth": {"username": "dialer", "password": _TRUNK_PASSWORD}},
}
_SIP_BODY: Final = {"session": {"model": "live-group"}, "transport": _SIP_TRANSPORT}
_SIP_ANSWER: Final = {"session": {"id": "live_upstream"}, "transport": {"type": "sip"}}


@pytest.mark.parametrize("status_code", [200, 201])
def test_a_sip_create_forwards_the_body_once_returns_a_token_and_keeps_the_status(
    live_client: TestClient, started_sidebands: list[_LiveSession], respx_mock: respx.MockRouter, status_code: int
) -> None:
    upstream: Final = respx_mock.post("https://api.openai.com/v1/live/sessions").mock(
        return_value=httpx.Response(status_code, json=_SIP_ANSWER)
    )

    response: Final = live_client.post("/v1/live/sessions", json=_SIP_BODY)

    expected_session: Final = _LiveSession(session_id="live_upstream", model="live-group", deployment_id="deployment-1")
    assert response.status_code == status_code, response.text
    assert upstream.call_count == 1
    assert json.loads(upstream.calls.last.request.content) == {
        "session": {"model": "gpt-live-1"},
        "transport": _SIP_TRANSPORT,
    }
    assert response.json()["transport"] == {"type": "sip"}
    assert _attachable_session(response.json()["session"]["id"], _OWNER) == expected_session
    assert started_sidebands == [expected_session]


@pytest.fixture
def sideband_cannot_join(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _does_not_join(_session: _LiveSession, _auth: UserAPIKeyAuth) -> bool:
        return False

    monkeypatch.setattr(live_endpoints, "_start_live_billing_sideband", _does_not_join)


@pytest.mark.usefixtures("sideband_cannot_join")
def test_a_sip_call_whose_billing_cannot_join_is_hung_up_once_and_answers_503(
    live_client: TestClient, respx_mock: respx.MockRouter
) -> None:
    create: Final = respx_mock.post("https://api.openai.com/v1/live/sessions").mock(
        return_value=httpx.Response(201, json=_SIP_ANSWER)
    )
    hangup: Final = respx_mock.post("https://api.openai.com/v1/live/sessions/live_upstream/hangup").mock(
        return_value=httpx.Response(200)
    )

    response: Final = live_client.post("/v1/live/sessions", json=_SIP_BODY)

    assert response.status_code == 503, response.text
    assert create.call_count == 1
    assert hangup.call_count == 1
    assert "session" not in response.json(), "a call that was ended leaves the caller nothing to control"


@pytest.mark.usefixtures("sideband_cannot_join")
@pytest.mark.parametrize(
    "hangup_outcome",
    [httpx.Response(500, json={"error": {"message": "failed"}}), httpx.ReadTimeout("timed out")],
    ids=["provider-error", "timeout"],
)
def test_a_sip_call_that_cannot_be_hung_up_returns_its_token_in_the_503(
    live_client: TestClient, respx_mock: respx.MockRouter, hangup_outcome: httpx.Response | Exception
) -> None:
    respx_mock.post("https://api.openai.com/v1/live/sessions").mock(return_value=httpx.Response(201, json=_SIP_ANSWER))
    hangup: Final = respx_mock.post("https://api.openai.com/v1/live/sessions/live_upstream/hangup").mock(
        side_effect=[hangup_outcome]
    )

    response: Final = live_client.post("/v1/live/sessions", json=_SIP_BODY)

    assert response.status_code == 503, response.text
    assert hangup.call_count == 1
    assert _attachable_session(response.json()["session"]["id"], _OWNER) == _LiveSession(
        session_id="live_upstream", model="live-group", deployment_id="deployment-1"
    )


@pytest.mark.usefixtures("sideband_cannot_join")
def test_a_webrtc_session_whose_billing_cannot_join_is_not_hung_up(live_client: TestClient) -> None:
    with respx.mock(assert_all_called=False) as provider:
        provider.post("https://api.openai.com/v1/live/sessions").mock(
            return_value=httpx.Response(
                201, json={"session": {"id": "live_upstream"}, "transport": {"type": "webrtc", "sdp": "v=0 answer"}}
            )
        )
        hangup: Final = provider.post("https://api.openai.com/v1/live/sessions/live_upstream/hangup").mock(
            return_value=httpx.Response(200)
        )
        response: Final = live_client.post("/v1/live/sessions", json=_CREATE_BODY)

    assert response.status_code == 503, response.text
    assert not hangup.called


@pytest.mark.parametrize("upstream_status", [400, 500])
def test_the_trunk_password_reaches_the_provider_and_no_log_or_error_body(
    live_client: TestClient,
    started_sidebands: list[_LiveSession],
    respx_mock: respx.MockRouter,
    caplog: pytest.LogCaptureFixture,
    upstream_status: int,
) -> None:
    upstream: Final = respx_mock.post("https://api.openai.com/v1/live/sessions").mock(
        return_value=httpx.Response(upstream_status, json={"error": {"message": f"bad trunk {_TRUNK_PASSWORD}"}})
    )
    loggers: Final = tuple(logging.getLogger(name) for name in ("LiteLLM", "LiteLLM Proxy", "LiteLLM Router"))
    propagated: Final = tuple(logger.propagate for logger in loggers)
    for logger in loggers:
        logger.propagate = True

    try:
        with caplog.at_level(logging.DEBUG):
            response: Final = live_client.post("/v1/live/sessions", json=_SIP_BODY)
    finally:
        for logger, value in zip(loggers, propagated, strict=True):
            logger.propagate = value

    assert _TRUNK_PASSWORD in upstream.calls.last.request.content.decode()
    assert response.status_code == upstream_status
    assert _TRUNK_PASSWORD not in response.text
    assert _TRUNK_PASSWORD not in caplog.text
    assert started_sidebands == []


def test_create_refuses_a_session_the_proxy_could_not_join_to_meter(
    live_client: TestClient, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    respx_mock.post("https://api.openai.com/v1/live/sessions").mock(
        return_value=httpx.Response(
            201, json={"session": {"id": "live_upstream"}, "transport": {"type": "webrtc", "sdp": "v=0 answer"}}
        )
    )

    async def _does_not_join(_session: _LiveSession, _auth: UserAPIKeyAuth) -> bool:
        return False

    monkeypatch.setattr(live_endpoints, "_start_live_billing_sideband", _does_not_join)

    response: Final = live_client.post("/v1/live/sessions", json=_CREATE_BODY)

    assert response.status_code == 503, response.text
    assert "v=0 answer" not in response.text, "without the SDP answer the browser cannot start an unmetered session"
    assert "could not join this session to meter it" in response.text


def test_create_starts_a_session_without_a_sideband_when_webrtc_metering_is_turned_off(
    live_client: TestClient, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    respx_mock.post("https://api.openai.com/v1/live/sessions").mock(
        return_value=httpx.Response(
            201, json={"session": {"id": "live_upstream"}, "transport": {"type": "webrtc", "sdp": "v=0 answer"}}
        )
    )
    started: Final[list[_LiveSession]] = []

    async def _record(session: _LiveSession, _auth: UserAPIKeyAuth) -> bool:
        started.append(session)
        return False

    monkeypatch.setattr(live_endpoints, "_start_live_billing_sideband", _record)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {"meter_webrtc_calls": False})

    response: Final = live_client.post("/v1/live/sessions", json=_CREATE_BODY)

    assert response.status_code == 201, response.text
    assert response.json()["transport"] == {"type": "webrtc", "sdp": "v=0 answer"}
    assert started == []
