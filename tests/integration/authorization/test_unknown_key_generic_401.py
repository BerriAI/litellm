import asyncio
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final, Literal

import anthropic
import httpx
import openai
import pytest
import websockets
from pydantic import BaseModel, ConfigDict, JsonValue
from websockets.exceptions import InvalidStatus

from tests.integration._support.client import Gateway, delete_key_if_present, eventually, object_value, string_value
from tests.integration._support.database import read_rows, write_rows
from tests.integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from tests.integration._support.provider import SharedProvider

MODEL: Final = "anthropic/claude-haiku-4-5"
GENERIC_MESSAGE: Final = "Authentication Error, Invalid API key provided."
CHAT: Final[Mapping[str, JsonValue]] = {"model": MODEL, "messages": [{"role": "user", "content": "unknown key"}]}
MESSAGES: Final[Mapping[str, JsonValue]] = {
    "model": MODEL,
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "unknown key"}],
}
RESPONSES: Final[Mapping[str, JsonValue]] = {"model": MODEL, "input": "unknown key"}
GEMINI: Final[Mapping[str, JsonValue]] = {"contents": [{"parts": [{"text": "unknown key"}]}]}

Credential = Literal["authorization", "x-api-key", "query"]
Side = Literal["gateway", "peer"]


class _ProxyError(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str
    type: str
    param: str
    code: str


class _ErrorResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    error: _ProxyError


GENERIC_ERROR: Final = _ProxyError(message=GENERIC_MESSAGE, type="token_not_found_in_db", param="key", code="401")


@dataclass(frozen=True, slots=True)
class _Route:
    method: Literal["GET", "POST"]
    path: str
    body: Mapping[str, JsonValue] | None
    credential: Credential


ROUTES: Final[Mapping[str, _Route]] = {
    "chat": _Route("POST", "/v1/chat/completions", CHAT, "authorization"),
    "chat_stream": _Route("POST", "/v1/chat/completions", {**CHAT, "stream": True}, "authorization"),
    "messages": _Route("POST", "/v1/messages", MESSAGES, "x-api-key"),
    "messages_stream": _Route("POST", "/v1/messages", {**MESSAGES, "stream": True}, "x-api-key"),
    "messages_bearer": _Route("POST", "/v1/messages", MESSAGES, "authorization"),
    "responses": _Route("POST", "/v1/responses", RESPONSES, "authorization"),
    "responses_stream": _Route("POST", "/v1/responses", {**RESPONSES, "stream": True}, "authorization"),
    "embeddings": _Route("POST", "/v1/embeddings", {"model": MODEL, "input": "unknown key"}, "authorization"),
    "completions": _Route("POST", "/v1/completions", {"model": MODEL, "prompt": "unknown key"}, "authorization"),
    "models": _Route("GET", "/v1/models", None, "authorization"),
    "anthropic_passthrough": _Route("POST", "/anthropic/v1/messages", MESSAGES, "x-api-key"),
    "gemini_passthrough": _Route("POST", "/gemini/v1beta/models/gemini-2.5-flash:generateContent", GEMINI, "query"),
    "key_info": _Route("GET", "/key/info", None, "authorization"),
    "user_info": _Route("GET", "/user/info", None, "authorization"),
}
BURST_ROUTES: Final = ("chat", "chat_stream", "messages", "messages_stream", "responses", "responses_stream")


def _unknown_key() -> str:
    return f"sk-{uuid.uuid4().hex}"


def _digest(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def _headers(credential: Credential, key: str) -> dict[str, str]:
    match credential:
        case "authorization":
            return {"Authorization": f"Bearer {key}"}
        case "x-api-key":
            return {"x-api-key": key, "anthropic-version": "2023-06-01"}
        case "query":
            return {}


def _params(credential: Credential, key: str) -> dict[str, str]:
    return {"key": key} if credential == "query" else {}


def _send(gateway: Gateway, route: _Route, key: str) -> httpx.Response:
    return gateway.client.request(
        route.method,
        route.path,
        json=None if route.body is None else dict(route.body),
        headers=_headers(route.credential, key),
        params=_params(route.credential, key),
    )


def _assert_generic_refusal(response: httpx.Response, *secrets: str) -> None:
    assert response.status_code == 401, response.text
    assert _ErrorResponse.model_validate_json(response.content).error == GENERIC_ERROR, response.text
    for secret in secrets:
        assert secret not in response.text, response.text


def _side(gateway: Gateway, peer: Gateway, side: Side) -> Gateway:
    return gateway if side == "gateway" else peer


@pytest.mark.parametrize("side", ("gateway", "peer"))
@pytest.mark.parametrize("route", tuple(ROUTES))
def test_an_unknown_key_gets_the_generic_401_without_reaching_the_provider(
    gateway: Gateway, peer: Gateway, provider: SharedProvider, route: str, side: Side
) -> None:
    key: Final = _unknown_key()
    _assert_generic_refusal(_send(_side(gateway, peer, side), ROUTES[route], key), key, key[-4:], _digest(key))
    assert provider.received() == ()


def test_the_openai_sdk_raises_authentication_error_carrying_the_generic_message(
    gateway: Gateway, provider: SharedProvider
) -> None:
    key: Final = _unknown_key()
    client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=key, max_retries=0)
    with pytest.raises(openai.AuthenticationError) as raised:
        client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "unknown key"}])
    assert raised.value.status_code == 401
    assert GENERIC_MESSAGE in raised.value.message
    assert _digest(key) not in raised.value.message
    assert provider.received() == ()


def test_the_anthropic_sdk_raises_authentication_error_carrying_the_generic_message(
    gateway: Gateway, provider: SharedProvider
) -> None:
    key: Final = _unknown_key()
    client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=key, max_retries=0)
    with pytest.raises(anthropic.AuthenticationError) as raised:
        client.messages.create(model=MODEL, max_tokens=16, messages=[{"role": "user", "content": "unknown key"}])
    assert raised.value.status_code == 401
    assert GENERIC_MESSAGE in raised.value.message
    assert _digest(key) not in raised.value.message
    assert provider.received() == ()


HOSTILE_KEYS: Final[Mapping[str, str]] = {"five_kb": f"sk-{'a' * 5120}", "bare_prefix": "sk-"}


@pytest.mark.parametrize("hostile", tuple(HOSTILE_KEYS))
def test_a_hostile_unknown_key_gets_the_generic_401_and_valid_keys_keep_working(
    gateway: Gateway, peer: Gateway, provider: SharedProvider, hostile: str
) -> None:
    key: Final = HOSTILE_KEYS[hostile]
    for candidate in (gateway, peer):
        _assert_generic_refusal(_send(candidate, ROUTES["chat"], key), _digest(key))
    with gateway.scenario() as scenario:
        valid: Final = scenario.key(models=[MODEL])
        listed: Final = tuple(candidate.request("GET", "/v1/models", key=valid) for candidate in (gateway, peer))
        assert [response.status_code for response in listed] == [200, 200], [response.text for response in listed]
    assert gateway.client.get("/health/liveliness").status_code == 200
    assert provider.received() == ()


def test_two_different_unknown_keys_in_both_headers_get_the_generic_401_naming_neither(
    gateway: Gateway, provider: SharedProvider
) -> None:
    bearer: Final = _unknown_key()
    api_key: Final = _unknown_key()
    headers: Final = {**_headers("authorization", bearer), **_headers("x-api-key", api_key)}
    for path, body in (("/v1/chat/completions", CHAT), ("/v1/messages", MESSAGES)):
        _assert_generic_refusal(
            gateway.client.post(path, json=dict(body), headers=headers),
            bearer,
            api_key,
            _digest(bearer),
            _digest(api_key),
        )
    assert provider.received() == ()


def test_the_same_unknown_key_sent_twice_across_both_proxies_gets_the_same_generic_401(
    gateway: Gateway, peer: Gateway, provider: SharedProvider
) -> None:
    key: Final = _unknown_key()
    responses: Final = tuple(_send(candidate, ROUTES["chat"], key) for candidate in (gateway, peer, gateway, peer))
    for response in responses:
        _assert_generic_refusal(response, key, _digest(key))
    assert len({response.content for response in responses}) == 1
    assert provider.received() == ()


def _refused_once_evicted(candidate: Gateway, key: str) -> httpx.Response:
    return eventually(
        lambda: httpx.get(
            f"{candidate.client.base_url}/v1/models", headers={"Authorization": f"Bearer {key}"}, trust_env=False
        ),
        lambda response: response.status_code == 401,
        seconds=75,
    )


def test_a_deleted_key_gets_the_generic_401_on_both_proxies(
    gateway: Gateway, peer: Gateway, provider: SharedProvider
) -> None:
    key: Final = string_value(gateway.post("/key/generate", {"models": [MODEL]})["key"])
    try:
        warmed: Final = tuple(candidate.request("GET", "/v1/models", key=key) for candidate in (gateway, peer))
        assert [response.status_code for response in warmed] == [200, 200], [response.text for response in warmed]
        gateway.post("/key/delete", {"keys": [key]})
        for refusal in (_refused_once_evicted(gateway, key), _refused_once_evicted(peer, key)):
            _assert_generic_refusal(refusal, key, _digest(key))
    finally:
        delete_key_if_present(gateway, key)
    assert provider.received() == ()


def test_an_unknown_key_failure_row_keeps_its_hash_and_records_the_generic_message(
    gateway: Gateway, provider: SharedProvider
) -> None:
    key: Final = _unknown_key()
    _assert_generic_refusal(_send(gateway, ROUTES["chat"], key), key)
    rows: Final = eventually(
        lambda: read_rows('SELECT status, metadata FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (_digest(key),)),
        lambda found: len(found) == 1,
        seconds=15,
    )
    assert rows[0]["status"] == "failure"
    error: Final = object_value(object_value(rows[0]["metadata"])["error_information"])
    assert (error["error_class"], error["error_code"], error["error_message"]) == (
        "KeyNotFoundError",
        "401",
        GENERIC_MESSAGE,
    )
    assert key not in json.dumps(rows[0]["metadata"])
    assert provider.received() == ()


def _control_error(response: httpx.Response) -> _ProxyError:
    assert response.status_code == 401, response.text
    return _ErrorResponse.model_validate_json(response.content).error


@pytest.mark.parametrize("side", ("gateway", "peer"))
def test_requests_without_an_sk_key_keep_their_own_401(gateway: Gateway, peer: Gateway, side: Side) -> None:
    client: Final = _side(gateway, peer, side).client
    assert _control_error(client.get("/v1/models")) == _ProxyError(
        message="Authentication Error, No api key passed in.", type="auth_error", param="None", code="401"
    )
    assert _control_error(client.get("/v1/models", headers={"Authorization": "Bearer not-a-real-key"})) == _ProxyError(
        message="LiteLLM Virtual Key expected. Received=not-****-key, expected to start with 'sk-'.",
        type="auth_error",
        param="None",
        code="401",
    )


def test_a_blocked_key_keeps_its_own_401_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key(models=[MODEL])
        gateway.post("/key/block", {"key": key})
        expected: Final = _ProxyError(
            message="Authentication Error, Key is blocked. Update via `/key/unblock` if you're an admin.",
            type="auth_error",
            param="None",
            code="401",
        )
        for candidate in (gateway, peer):
            assert _control_error(_refused_once_evicted(candidate, key)) == expected


def test_an_expired_key_keeps_its_own_401_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key(models=[MODEL])
        write_rows(
            "UPDATE \"LiteLLM_VerificationToken\" SET expires = now() - interval '1 hour' WHERE token=%s",
            (_digest(key),),
        )
        errors: Final = tuple(
            _control_error(candidate.request("GET", "/v1/models", key=key)) for candidate in (gateway, peer)
        )
        abbreviations: Final = {f"sk-...{key[-4:]}", f"sk-...{_digest(key)[-4:]}"}
        for error in errors:
            assert error.message.startswith("Authentication Error - Expired Key. Key Expiry time "), error
            assert (error.type, error.code) == ("expired_key", "401"), error
            assert error.param in abbreviations, error


def test_the_mcp_endpoint_keeps_its_own_401_for_an_unknown_key(gateway: Gateway, peer: Gateway) -> None:
    key: Final = _unknown_key()
    initialize: Final = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "audit", "version": "1"},
        },
    }
    headers: Final = {"Authorization": f"Bearer {key}", "Accept": "application/json, text/event-stream"}
    responses: Final = tuple(
        candidate.client.post("/mcp/", json=initialize, headers=headers) for candidate in (gateway, peer)
    )
    for response in responses:
        assert response.status_code == 401, response.text
        assert response.json() == {
            "detail": {
                "error": "authentication_required",
                "message": "Authenticate with the gateway to use the MCP endpoint.",
            }
        }


async def _realtime_refusal(candidate: Gateway, key: str) -> InvalidStatus:
    url: Final = str(candidate.client.base_url).rstrip("/").replace("http://", "ws://")
    with pytest.raises(InvalidStatus) as refusal:
        async with websockets.connect(
            f"{url}/v1/realtime?model={MODEL}", additional_headers={"Authorization": f"Bearer {key}"}
        ):
            pass
    return refusal.value


async def test_the_realtime_upgrade_refuses_an_unknown_key_on_both_proxies_without_echoing_it(
    gateway: Gateway, peer: Gateway, provider: SharedProvider
) -> None:
    key: Final = _unknown_key()
    refusals: Final = await asyncio.gather(*(_realtime_refusal(candidate, key) for candidate in (gateway, peer)))
    assert [refusal.response.status_code for refusal in refusals] == [403, 403]
    seen: Final = " ".join(
        f"{refusal.response.reason_phrase} {refusal.response.headers} {refusal.response.body!r}" for refusal in refusals
    )
    assert [secret for secret in (key, f"sk-...{key[-4:]}", _digest(key)) if secret in seen] == []
    assert provider.received() == ()


async def _send_async(client: httpx.AsyncClient, route: _Route, key: str) -> httpx.Response:
    return await client.request(
        route.method,
        route.path,
        json=None if route.body is None else dict(route.body),
        headers=_headers(route.credential, key),
    )


async def _burst(owned: OwnedProxy, keys: tuple[str, ...]) -> tuple[httpx.Response, ...]:
    async with httpx.AsyncClient(base_url=str(owned.gateway.client.base_url), timeout=60, trust_env=False) as client:
        return tuple(
            await asyncio.gather(
                *(
                    _send_async(client, ROUTES[BURST_ROUTES[index % len(BURST_ROUTES)]], key)
                    for index, key in enumerate(keys)
                )
            )
        )


def _log_after(owned: OwnedProxy, offset: int) -> str:
    return owned.log.read_bytes()[offset:].decode(errors="replace")


def _console_line(key: str) -> str:
    return f"key={_digest(key)}, not found in db"


@pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120))
async def test_a_burst_across_a_worker_kill_answers_every_unknown_key_with_the_generic_401_and_logs_its_hash_once(
    gateway: Gateway, tmp_path: Path
) -> None:
    keys: Final = tuple(_unknown_key() for _ in range(40))
    with owned_proxy_process(gateway, tmp_path, {}, workers=2) as owned:
        victim, _ = sorted(eventually(owned.worker_pids, lambda pids: len(pids) == 2))
        offset: Final = owned.log.stat().st_size
        with owned.replacing_worker(victim) as kill:
            kill.freeze()
            during, _ = await asyncio.gather(_burst(owned, keys[:20]), asyncio.to_thread(kill))
        after: Final = await _burst(owned, keys[20:])
        window: Final = await asyncio.to_thread(
            eventually,
            lambda: _log_after(owned, offset),
            lambda text: all(_console_line(key) in text for key in keys),
            30,
        )
    for response, key in zip((*during, *after), keys, strict=True):
        _assert_generic_refusal(response, key, _digest(key))
    assert [window.count(_console_line(key)) for key in keys] == [1] * len(keys)
    assert [key for key in keys if key in window] == []
