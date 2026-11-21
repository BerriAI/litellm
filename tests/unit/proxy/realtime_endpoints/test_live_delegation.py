import json
from typing import Final

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.realtime_endpoints import live_endpoints
from litellm.proxy.realtime_endpoints.live_delegation import LiveDelegation, live_delegation, live_delegation_denial
from litellm.proxy.realtime_endpoints.live_endpoints import _LiveSession

_FILE_SEARCH: Final = {"type": "file_search", "vector_store_ids": ["vs_granted", "vs_other"]}


def _session(responses: dict[str, object]) -> dict[str, object]:
    return {"model": "live-group", "delegation": {"type": "responses", "responses": responses}}


@pytest.mark.parametrize(
    ("session", "delegation"),
    [
        ({"model": "live-group"}, LiveDelegation(models=(), vector_store_ids=())),
        ({"model": "live-group", "delegation": "responses"}, LiveDelegation(models=(), vector_store_ids=())),
        (_session({"model": "text-group"}), LiveDelegation(models=("text-group",), vector_store_ids=())),
        (
            _session({"model": "text-group", "tools": [{"type": "function", "name": "lookup"}, _FILE_SEARCH]}),
            LiveDelegation(models=("text-group",), vector_store_ids=("vs_granted", "vs_other")),
        ),
        (
            {"model": "live-group", "delegation": {"type": "agents", "agents": [{"backend": {"model": "text-group"}}]}},
            LiveDelegation(models=("text-group",), vector_store_ids=()),
        ),
    ],
    ids=["no-delegation", "not-a-mapping", "model-only", "model-and-file-search", "a-shape-not-seen-before"],
)
def test_live_delegation_names_every_model_and_vector_store_the_session_hands_work_to(
    session: dict[str, object], delegation: LiveDelegation
) -> None:
    assert live_delegation(session) == delegation


def _key(
    models: list[str], vector_stores: list[str] | None = None, allowed_tools: list[str] | None = None
) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        token="owner-key-hash",
        models=models,
        metadata={} if allowed_tools is None else {"allowed_tools": allowed_tools},
        object_permission=None
        if vector_stores is None
        else LiteLLM_ObjectPermissionTable(object_permission_id="permission-1", vector_stores=vector_stores),
    )


@pytest.fixture
def started_sidebands(monkeypatch: pytest.MonkeyPatch) -> list[_LiveSession]:
    started: Final[list[_LiveSession]] = []  # mutable-ok: collects each billing sideband the route starts

    async def _joins(session: _LiveSession, _auth: UserAPIKeyAuth) -> bool:
        started.append(session)
        return True

    monkeypatch.setattr(live_endpoints, "_start_live_billing_sideband", _joins)
    return started


@pytest.fixture
def proxy_with_live_and_text_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-live-session-test-salt")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    model_list: Final = [  # mutable-ok: Router takes a list of deployments
        {
            "model_name": "live-group",
            "litellm_params": {"model": "openai/gpt-live-1", "api_key": "sk-upstream"},
            "model_info": {"id": "deployment-1"},
        },
        {
            "model_name": "text-group",
            "litellm_params": {"model": "openai/gpt-6-luna", "api_key": "sk-upstream"},
            "model_info": {"id": "deployment-2"},
        },
    ]
    monkeypatch.setattr(proxy_server, "llm_router", litellm.Router(model_list=model_list))
    monkeypatch.setattr(proxy_server, "llm_model_list", model_list)


def _create(key: UserAPIKeyAuth, session: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> httpx.Response:
    from litellm.proxy import proxy_server

    monkeypatch.setitem(proxy_server.app.dependency_overrides, user_api_key_auth, lambda: key)
    return TestClient(proxy_server.app, raise_server_exceptions=False).post(
        "/v1/live/sessions", json={"session": session, "transport": {"type": "webrtc", "sdp": "v=0 offer"}}
    )


@pytest.mark.usefixtures("proxy_with_live_and_text_groups")
@pytest.mark.parametrize(
    ("key", "responses"),
    [
        (_key(["live-group"]), {"model": "text-group"}),
        (_key(["live-group", "text-group"], ["vs_granted"]), {"model": "text-group", "tools": [_FILE_SEARCH]}),
        (
            _key(["live-group", "text-group"], allowed_tools=["lookup"]),
            {
                "model": "text-group",
                "tools": [{"type": "function", "name": "lookup"}, {"type": "function", "name": "wire"}],
            },
        ),
    ],
    ids=["model-the-key-cannot-call", "vector-store-the-key-was-not-granted", "tool-outside-the-key-allowlist"],
)
def test_create_refuses_a_session_that_delegates_to_what_the_key_cannot_use(
    key: UserAPIKeyAuth,
    responses: dict[str, object],
    started_sidebands: list[_LiveSession],
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream: Final = respx_mock.post("https://api.openai.com/v1/live/sessions").mock(
        return_value=httpx.Response(201, json={"session": {"id": "live_upstream"}})
    )

    response: Final = _create(key, _session(responses), monkeypatch)

    assert response.status_code in (401, 403), response.text
    assert upstream.call_count == 0, "the provider session must not exist when the key may not use what it delegates to"
    assert started_sidebands == []


@pytest.mark.usefixtures("proxy_with_live_and_text_groups")
def test_create_forwards_a_delegation_the_key_may_use(
    started_sidebands: list[_LiveSession], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream: Final = respx_mock.post("https://api.openai.com/v1/live/sessions").mock(
        return_value=httpx.Response(
            201, json={"session": {"id": "live_upstream"}, "transport": {"type": "webrtc", "sdp": "v=0 answer"}}
        )
    )
    responses: Final = {
        "model": "text-group",
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_granted"]}, {"type": "function", "name": "lookup"}],
    }

    response: Final = _create(
        _key(["live-group", "text-group"], ["vs_granted"], ["lookup", "file_search"]), _session(responses), monkeypatch
    )

    assert response.status_code == 201, response.text
    assert json.loads(upstream.calls.last.request.content)["session"] == {
        "model": "gpt-live-1",
        "delegation": {"type": "responses", "responses": responses},
    }
    assert len(started_sidebands) == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("proxy_with_live_and_text_groups")
@pytest.mark.parametrize(
    ("client_frame", "denied"),
    [
        ({"type": "session.update", "session": _session({"model": "text-group"})}, True),
        ({"type": "session.update", "session": _session({"model": "live-group"})}, False),
        ({"type": "session.update", "session": {"instructions": "Explain delegation"}}, False),
        ({"type": "input_audio.append", "audio": "delegation"}, False),
    ],
    ids=["delegates-to-a-model-the-key-cannot-call", "delegates-to-a-model-the-key-can-call", "no-delegation", "audio"],
)
async def test_a_client_frame_is_denied_only_when_it_delegates_to_what_the_key_cannot_use(
    client_frame: dict[str, object], denied: bool
) -> None:
    denial: Final = await live_delegation_denial(json.dumps(client_frame), _key(["live-group"]))

    assert (denial is not None) is denied, denial


@pytest.mark.asyncio
async def test_a_client_frame_that_spells_delegation_with_a_json_escape_is_still_checked() -> None:
    client_frame: Final = json.dumps({"type": "session.update", "session": _session({"model": "other-group"})}).replace(
        '"delegation"', '"deleg\\u0061tion"'
    )

    assert "delegation" not in client_frame
    assert await live_delegation_denial(client_frame, _key(["live-group"])) is not None
