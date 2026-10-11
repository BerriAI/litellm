import asyncio
import hashlib
import itertools
import json
import re
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import anthropic
import httpx
import openai
import pytest
import yaml
from integration._support import device_login as dl
from integration._support import responses_vendor as rv
from integration._support.client import eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Request
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(300)

Provider: TypeAlias = Literal["chatgpt", "copilot"]
Endpoint: TypeAlias = Literal["chat", "responses", "messages"]
Client: TypeAlias = Literal["sdk", "async_sdk", "httpx"]

_PROVIDERS: Final[tuple[Provider, ...]] = ("chatgpt", "copilot")
_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "responses", "messages")
_CLIENTS: Final[tuple[Client, ...]] = ("sdk", "async_sdk", "httpx")
_MODELS: Final[Mapping[tuple[Provider, Endpoint], str]] = MappingProxyType(
    {
        ("chatgpt", "chat"): "chatgpt/gpt-5.6-terra",
        ("chatgpt", "responses"): "chatgpt/gpt-5.6-terra",
        ("chatgpt", "messages"): "chatgpt/gpt-5.6-terra",
        ("copilot", "chat"): "github_copilot/gpt-5.4",
        ("copilot", "responses"): "github_copilot/gpt-5.4",
        ("copilot", "messages"): "github_copilot/claude-sonnet-5.5",
    }
)
_REFUSALS: Final[Mapping[Provider, str]] = MappingProxyType(
    {"chatgpt": dl.CHATGPT_REFUSAL, "copilot": dl.COPILOT_REFUSAL}
)
_STORED_BEARERS: Final[Mapping[Provider, str]] = MappingProxyType(
    {"chatgpt": dl.CHATGPT_STORED, "copilot": dl.COPILOT_STORED_KEY}
)
_CONTROL_MODEL: Final = "device-login-control"
_EXTRA: Final[Mapping[str, JsonValue]] = MappingProxyType({"num_retries": 0, "cache": {"no-cache": True}})
_CALL_ID: Final = "x-litellm-call-id"
_CLIENT_SECONDS: Final = 60


@dataclass(frozen=True, slots=True)
class _Cell:
    provider: Provider
    endpoint: Endpoint
    stream: bool
    client: Client

    def name(self) -> str:
        return f"{self.provider}-{self.endpoint}-{'stream' if self.stream else 'unary'}-{self.client}"


_CELLS: Final = tuple(
    _Cell(provider, endpoint, stream, client)
    for provider, endpoint, stream, client in itertools.product(_PROVIDERS, _ENDPOINTS, (False, True), _CLIENTS)
)


@dataclass(frozen=True, slots=True)
class _Call:
    base_url: str
    key: str
    model: str
    endpoint: Endpoint
    stream: bool
    client: Client
    marker: str
    served: SimpleQueue[str]

    def prompt(self) -> str:
        return f"Reply to marker-{self.marker}"


@dataclass(frozen=True, slots=True)
class _Served:
    status: int
    text: str
    call_id: str


@dataclass(frozen=True, slots=True)
class _Rig:
    owned: OwnedProxy
    peers: dl.Peers
    chatgpt_dir: Path
    copilot_dir: Path

    def base_url(self) -> str:
        return str(self.owned.gateway.client.base_url).rstrip("/")


@dataclass(frozen=True, slots=True)
class _Scene:
    rig: _Rig
    key: str
    log_offset: int
    served: SimpleQueue[str]

    def call(self, cell: _Cell) -> _Call:
        return _Call(
            self.rig.base_url(),
            self.key,
            _MODELS[cell.provider, cell.endpoint],
            cell.endpoint,
            cell.stream,
            cell.client,
            uuid.uuid4().hex,
            self.served,
        )

    def control(self, stream: bool) -> _Call:
        return _Call(
            self.rig.base_url(), self.key, _CONTROL_MODEL, "chat", stream, "httpx", uuid.uuid4().hex, self.served
        )

    def post(self, path: str, body: Mapping[str, JsonValue], *, as_admin: bool = False) -> _Served:
        key: Final = self.rig.owned.gateway.key if as_admin else self.key
        with httpx.Client(base_url=self.rig.base_url(), timeout=_CLIENT_SECONDS, trust_env=False) as client:
            response: Final = client.post(
                path, json=dict(body), headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"}
            )
        return _Served(response.status_code, response.text, response.headers.get(_CALL_ID, ""))

    def token_files(self) -> tuple[str, ...]:
        return tuple(sorted(path.name for path in (*self.rig.chatgpt_dir.iterdir(), *self.rig.copilot_dir.iterdir())))

    def log_tail(self) -> str:
        with self.rig.owned.log.open("rb") as log:
            log.seek(self.log_offset)
            return log.read().decode(errors="replace")

    def store_valid_token(self, provider: Provider) -> None:
        if provider == "chatgpt":
            dl.write_chatgpt(self.rig.chatgpt_dir, dl.chatgpt_record(dl.CHATGPT_STORED))
            return
        dl.write_copilot_key(self.rig.copilot_dir, dl.COPILOT_STORED_KEY, self.rig.peers.api.url)


def _config(peers: dl.Peers, directory: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        {"model_name": "*", "litellm_params": {"model": "*"}},
        {
            "model_name": _CONTROL_MODEL,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_base": f"{peers.api.url}/v1",
                "api_key": dl.CONTROL_KEY,
            },
        },
    ]
    path: Final = directory / "device-login-guard.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("device-login-guard")
    chatgpt_dir: Final = directory / "chatgpt"
    copilot_dir: Final = directory / "copilot"
    chatgpt_dir.mkdir()
    copilot_dir.mkdir()
    with (
        gateway_from_environment() as gateway,
        dl.device_login_peers(directory) as peers,
        owned_proxy_process(
            gateway,
            directory,
            {
                **peers.environment(),
                "CHATGPT_TOKEN_DIR": str(chatgpt_dir),
                "GITHUB_COPILOT_TOKEN_DIR": str(copilot_dir),
            },
            config=_config(peers, directory),
            workers=2,
            remove_environment=("REQUEST_TIMEOUT",),
        ) as owned,
    ):
        yield _Rig(owned, peers, chatgpt_dir, copilot_dir)


@pytest.fixture
def scene(rig: _Rig) -> Iterator[_Scene]:
    rig.peers.reset()
    dl.clear_tokens(rig.chatgpt_dir, rig.copilot_dir)
    served: Final[SimpleQueue[str]] = SimpleQueue()
    with rig.owned.gateway.scenario() as scenario:
        key: Final = scenario.key()
        yield _Scene(rig, key, rig.owned.log.stat().st_size, served)
        _every_served_call_logged(key, served)
    rig.peers.reset()
    dl.clear_tokens(rig.chatgpt_dir, rig.copilot_dir)


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "responses":
            return "/v1/responses"
        case "messages":
            return "/v1/messages"


def _raw_body(call: _Call) -> Mapping[str, JsonValue]:
    common: Final[Mapping[str, JsonValue]] = MappingProxyType({"model": call.model, "stream": call.stream, **_EXTRA})
    match call.endpoint:
        case "chat":
            return {**common, "messages": [{"role": "user", "content": call.prompt()}]}
        case "responses":
            return {**common, "input": call.prompt()}
        case "messages":
            return {**common, "max_tokens": 64, "messages": [{"role": "user", "content": call.prompt()}]}


def _httpx(call: _Call, seconds: float = _CLIENT_SECONDS) -> _Served:
    with (
        httpx.Client(base_url=call.base_url, timeout=seconds, trust_env=False) as client,
        client.stream(
            "POST",
            _path(call.endpoint),
            json=_raw_body(call),
            headers={"Authorization": f"Bearer {call.key}", "anthropic-version": "2023-06-01"},
        ) as response,
    ):
        return _record(
            call, _Served(response.status_code, response.read().decode(), response.headers.get(_CALL_ID, ""))
        )


def _refused(error: openai.APIStatusError | anthropic.APIStatusError) -> _Served:
    return _Served(error.status_code, error.response.text, error.response.headers.get(_CALL_ID, ""))


def _sdk(call: _Call) -> _Served:
    try:
        if call.endpoint == "messages":
            with (
                anthropic.Anthropic(
                    base_url=call.base_url, api_key=call.key, max_retries=0, timeout=_CLIENT_SECONDS
                ) as claude,
                claude.messages.with_streaming_response.create(
                    model=call.model,
                    max_tokens=64,
                    messages=[{"role": "user", "content": call.prompt()}],
                    stream=call.stream,
                    extra_body=dict(_EXTRA),
                ) as message,
            ):
                return _Served(message.status_code, message.text(), message.headers.get(_CALL_ID, ""))
        with openai.OpenAI(
            base_url=f"{call.base_url}/v1", api_key=call.key, max_retries=0, timeout=_CLIENT_SECONDS
        ) as client:
            if call.endpoint == "chat":
                with client.chat.completions.with_streaming_response.create(
                    model=call.model,
                    messages=[{"role": "user", "content": call.prompt()}],
                    stream=call.stream,
                    extra_body=dict(_EXTRA),
                ) as completion:
                    return _Served(completion.status_code, completion.text(), completion.headers.get(_CALL_ID, ""))
            with client.responses.with_streaming_response.create(
                model=call.model, input=call.prompt(), stream=call.stream, extra_body=dict(_EXTRA)
            ) as created:
                return _Served(created.status_code, created.text(), created.headers.get(_CALL_ID, ""))
    except (openai.APIStatusError, anthropic.APIStatusError) as error:
        return _refused(error)


async def _async_sdk(call: _Call) -> _Served:
    try:
        if call.endpoint == "messages":
            async with (
                anthropic.AsyncAnthropic(
                    base_url=call.base_url, api_key=call.key, max_retries=0, timeout=_CLIENT_SECONDS
                ) as claude,
                claude.messages.with_streaming_response.create(
                    model=call.model,
                    max_tokens=64,
                    messages=[{"role": "user", "content": call.prompt()}],
                    stream=call.stream,
                    extra_body=dict(_EXTRA),
                ) as message,
            ):
                return _Served(message.status_code, await message.text(), message.headers.get(_CALL_ID, ""))
        async with openai.AsyncOpenAI(
            base_url=f"{call.base_url}/v1", api_key=call.key, max_retries=0, timeout=_CLIENT_SECONDS
        ) as client:
            if call.endpoint == "chat":
                async with client.chat.completions.with_streaming_response.create(
                    model=call.model,
                    messages=[{"role": "user", "content": call.prompt()}],
                    stream=call.stream,
                    extra_body=dict(_EXTRA),
                ) as completion:
                    return _Served(
                        completion.status_code, await completion.text(), completion.headers.get(_CALL_ID, "")
                    )
            async with client.responses.with_streaming_response.create(
                model=call.model, input=call.prompt(), stream=call.stream, extra_body=dict(_EXTRA)
            ) as created:
                return _Served(created.status_code, await created.text(), created.headers.get(_CALL_ID, ""))
    except (openai.APIStatusError, anthropic.APIStatusError) as error:
        return _refused(error)


def _serve(call: _Call) -> _Served:
    match call.client:
        case "sdk":
            return _record(call, _sdk(call))
        case "async_sdk":
            return _record(call, asyncio.run(_async_sdk(call)))
        case "httpx":
            return _httpx(call)


def _error_message(text: str) -> str:
    error: Final = rv.JSON_OBJECT.validate_python(rv.JSON_OBJECT.validate_json(text)["error"])
    return str(error["message"])


def _spend_rows(key: str) -> Sequence[Mapping[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT request_id, status, model FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
            (hashlib.sha256(key.encode()).hexdigest(),),
        ),
        lambda rows: len(rows) >= 1,
        seconds=40,
    )


def _record(call: _Call, served: _Served) -> _Served:
    call.served.put(served.call_id)
    return served


def _every_served_call_logged(key: str, served: SimpleQueue[str]) -> None:
    served_calls: Final = tuple(served.get_nowait() for _ in range(served.qsize()))
    if not served_calls:
        return
    eventually(
        lambda: read_rows(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
            (hashlib.sha256(key.encode()).hexdigest(),),
        ),
        lambda rows: len(rows) >= len(served_calls),
        seconds=40,
    )


def _failure_logged(served: _Served) -> None:
    rows: Final = eventually(
        lambda: read_rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (served.call_id,)),
        lambda found: len(found) >= 1,
        seconds=40,
    )
    assert [str(row["status"]) for row in rows] == ["failure"], rows


def _carrying(received: tuple[Request, ...], marker: str) -> tuple[Request, ...]:
    return tuple(request for request in received if marker.encode() in request.body)


def _describe(received: tuple[Request, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((request.method, request.target) for request in received)


@pytest.mark.parametrize("cell", _CELLS, ids=_Cell.name)
def test_missing_token_is_refused_with_the_login_instructions_before_any_auth_host_call(
    scene: _Scene, cell: _Cell
) -> None:
    call: Final = scene.call(cell)
    served: Final = _serve(call)
    assert served.status == 400, served.text
    assert _REFUSALS[cell.provider] in _error_message(served.text), served.text
    assert scene.rig.peers.auth_connections() == ()
    assert _describe(scene.rig.peers.api.drain()) == ()
    (row,) = _spend_rows(scene.key)
    assert (row["request_id"], row["status"]) == (served.call_id, "failure"), row


@pytest.mark.parametrize("cell", _CELLS, ids=_Cell.name)
def test_stored_token_serves_the_request_without_an_auth_host_call(scene: _Scene, cell: _Cell) -> None:
    scene.store_valid_token(cell.provider)
    call: Final = scene.call(cell)
    served: Final = _serve(call)
    assert served.status == 200, served.text
    assert set(rv.MARKER.findall(served.text)) == {call.marker}, served.text
    received: Final = scene.rig.peers.api.drain()
    (forwarded,) = _carrying(received, call.marker)
    assert dl.bearer(forwarded) == _STORED_BEARERS[cell.provider], _describe(received)
    assert scene.rig.peers.auth_connections() == ()
    (row,) = _spend_rows(scene.key)
    assert row["status"] == "success", row


def test_expired_chatgpt_token_is_refreshed_once_and_the_request_is_served(scene: _Scene) -> None:
    dl.write_chatgpt(scene.rig.chatgpt_dir, dl.chatgpt_record(dl.CHATGPT_EXPIRED, refresh_token=dl.GOOD_REFRESH))
    call: Final = scene.call(_Cell("chatgpt", "responses", False, "httpx"))
    served: Final = _serve(call)
    assert served.status == 200, served.text
    assert set(rv.MARKER.findall(served.text)) == {call.marker}, served.text
    (refresh,) = scene.rig.peers.auth.drain()
    assert (refresh.method, refresh.target, refresh.headers.get("host")) == (
        "POST",
        "/oauth/token",
        dl.CHATGPT_AUTH_HOST,
    )
    sent: Final = rv.JSON_OBJECT.validate_json(refresh.body)
    assert (sent["grant_type"], sent["refresh_token"]) == ("refresh_token", dl.GOOD_REFRESH), sent
    (forwarded,) = _carrying(scene.rig.peers.api.drain(), call.marker)
    assert dl.bearer(forwarded) == dl.CHATGPT_REFRESHED
    stored: Final = json.loads((scene.rig.chatgpt_dir / "auth.json").read_text())
    assert stored["access_token"] == dl.CHATGPT_REFRESHED, sorted(stored)


def test_revoked_chatgpt_refresh_token_is_refused_after_the_refresh_call_alone(scene: _Scene) -> None:
    dl.write_chatgpt(scene.rig.chatgpt_dir, dl.chatgpt_record(dl.CHATGPT_EXPIRED, refresh_token=dl.REVOKED_REFRESH))
    served: Final = _serve(scene.call(_Cell("chatgpt", "chat", False, "httpx")))
    assert served.status == 400, served.text
    assert dl.CHATGPT_REFUSAL in _error_message(served.text), served.text
    _failure_logged(served)
    reached: Final = scene.rig.peers.auth.drain()
    assert {(request.method, request.target) for request in reached} == {("POST", "/oauth/token")}, _describe(reached)
    assert _describe(scene.rig.peers.api.drain()) == ()


@dataclass(frozen=True, slots=True)
class _Route:
    name: str
    provider: Provider
    path: str
    body: Mapping[str, JsonValue]

    def label(self) -> str:
        return self.name


_ROUTES: Final = (
    _Route("chatgpt-embeddings", "chatgpt", "/v1/embeddings", {"model": "chatgpt/gpt-5.6-terra", "input": "login"}),
    _Route("chatgpt-completions", "chatgpt", "/v1/completions", {"model": "chatgpt/gpt-5.6-terra", "prompt": "login"}),
    _Route(
        "chatgpt-images", "chatgpt", "/v1/images/generations", {"model": "chatgpt/gpt-5.6-terra", "prompt": "login"}
    ),
    _Route(
        "copilot-embeddings",
        "copilot",
        "/v1/embeddings",
        {"model": "github_copilot/text-embedding-3-small", "input": "login"},
    ),
    _Route("copilot-completions", "copilot", "/v1/completions", {"model": "github_copilot/gpt-5.4", "prompt": "login"}),
    _Route(
        "copilot-images", "copilot", "/v1/images/generations", {"model": "github_copilot/gpt-5.4", "prompt": "login"}
    ),
)


@pytest.mark.parametrize("route", _ROUTES, ids=_Route.label)
def test_missing_token_is_refused_on_the_other_model_routes(scene: _Scene, route: _Route) -> None:
    served: Final = scene.post(route.path, {**route.body, **_EXTRA})
    assert served.status == 400, served.text
    assert _REFUSALS[route.provider] in _error_message(served.text), served.text
    assert scene.rig.peers.auth_connections() == ()
    assert _describe(scene.rig.peers.api.drain()) == ()


def test_copilot_embeddings_are_served_with_the_stored_api_key(scene: _Scene) -> None:
    scene.store_valid_token("copilot")
    served: Final = scene.post(
        "/v1/embeddings", {"model": "github_copilot/text-embedding-3-small", "input": "login", **_EXTRA}
    )
    assert served.status == 200, served.text
    (item,) = rv.ITEMS.validate_python(rv.JSON_OBJECT.validate_json(served.text)["data"])
    assert item["embedding"] == [0.25, 0.5, 0.75], served.text
    (forwarded,) = scene.rig.peers.api.drain()
    assert (forwarded.method, dl.bearer(forwarded)) == ("POST", dl.COPILOT_STORED_KEY), forwarded.target
    assert forwarded.target.endswith("/embeddings"), forwarded.target
    assert scene.rig.peers.auth_connections() == ()


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_connection_test_reports_the_refusal_instead_of_starting_a_login(scene: _Scene, provider: Provider) -> None:
    served: Final = scene.post(
        "/health/test_connection",
        {"litellm_params": {"model": _MODELS[provider, "chat"]}, "mode": "chat"},
        as_admin=True,
    )
    assert served.status == 200, served.text
    report: Final = rv.JSON_OBJECT.validate_json(served.text)
    assert report["status"] == "error", served.text
    assert _REFUSALS[provider] in str(rv.JSON_OBJECT.validate_python(report["result"])["error"]), served.text
    assert scene.rig.peers.auth_connections() == ()


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_count_tokens_answers_locally_without_a_login(scene: _Scene, provider: Provider) -> None:
    served: Final = scene.post(
        "/v1/messages/count_tokens",
        {"model": _MODELS[provider, "messages"], "messages": [{"role": "user", "content": "count these tokens"}]},
    )
    assert served.status == 200, served.text
    counted: Final = rv.JSON_OBJECT.validate_json(served.text)["input_tokens"]
    assert isinstance(counted, int) and counted > 0, served.text
    assert scene.rig.peers.auth_connections() == ()


def test_health_check_still_answers_without_a_login(scene: _Scene) -> None:
    response: Final = scene.rig.owned.gateway.request("GET", "/health", None)
    assert response.status_code == 200, response.text
    health: Final = rv.JSON_OBJECT.validate_json(response.text)
    healthy: Final = rv.ITEMS.validate_python(health["healthy_endpoints"])
    assert [endpoint["model"] for endpoint in healthy] == ["openai/gpt-4o-mini"], response.text
    assert scene.rig.peers.auth_connections() == ()
    tail: Final = scene.log_tail()
    assert dl.CHATGPT_USER_CODE not in tail and dl.COPILOT_USER_CODE not in tail, tail[-2000:]


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("provider", _PROVIDERS)
def test_an_auth_host_ready_to_grant_a_login_is_never_asked_and_no_code_reaches_the_log(
    scene: _Scene, provider: Provider, endpoint: Endpoint
) -> None:
    scene.rig.peers.switches.grant.set()
    served: Final = _serve(scene.call(_Cell(provider, endpoint, False, "httpx")))
    assert served.status == 400, served.text
    assert _REFUSALS[provider] in _error_message(served.text), served.text
    assert scene.rig.peers.auth_connections() == ()
    assert _describe(scene.rig.peers.auth.drain()) == ()
    tail: Final = scene.log_tail()
    assert dl.CHATGPT_USER_CODE not in tail and dl.COPILOT_USER_CODE not in tail, tail[-2000:]
    assert scene.token_files() == ()


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("provider", _PROVIDERS)
def test_stored_token_the_provider_rejects_surfaces_its_401_without_a_login(
    scene: _Scene, provider: Provider, endpoint: Endpoint
) -> None:
    if provider == "chatgpt":
        dl.write_chatgpt(scene.rig.chatgpt_dir, dl.chatgpt_record(dl.CHATGPT_REJECTED))
    else:
        dl.write_copilot_key(scene.rig.copilot_dir, dl.COPILOT_REJECTED_KEY, scene.rig.peers.api.url)
    call: Final = scene.call(_Cell(provider, endpoint, False, "httpx"))
    served: Final = _serve(call)
    assert served.status == 401, served.text
    assert dl.REJECTED_DETAIL in served.text, served.text
    received: Final = scene.rig.peers.api.drain()
    assert {dl.bearer(request) for request in _carrying(received, call.marker)} == {
        dl.CHATGPT_REJECTED if provider == "chatgpt" else dl.COPILOT_REJECTED_KEY
    }, _describe(received)
    assert scene.rig.peers.auth_connections() == ()


@pytest.mark.parametrize("stale_key", (False, True), ids=("missing-api-key", "expired-api-key"))
def test_copilot_api_key_is_minted_once_from_the_stored_access_token(scene: _Scene, stale_key: bool) -> None:
    dl.write_copilot_access(scene.rig.copilot_dir, dl.COPILOT_ACCESS)
    if stale_key:
        dl.write_copilot_key(scene.rig.copilot_dir, dl.COPILOT_REJECTED_KEY, scene.rig.peers.api.url, -3600)
    call: Final = scene.call(_Cell("copilot", "chat", False, "httpx"))
    served: Final = _serve(call)
    assert served.status == 200, served.text
    assert set(rv.MARKER.findall(served.text)) == {call.marker}, served.text
    (minted,) = scene.rig.peers.auth.drain()
    assert (minted.method, minted.target, minted.headers.get("host"), minted.headers.get("authorization")) == (
        "GET",
        "/copilot_internal/v2/token",
        dl.GITHUB_API_HOST,
        f"token {dl.COPILOT_ACCESS}",
    )
    (forwarded,) = _carrying(scene.rig.peers.api.drain(), call.marker)
    assert dl.bearer(forwarded) == dl.COPILOT_MINTED_KEY
    assert json.loads((scene.rig.copilot_dir / "api-key.json").read_text())["token"] == dl.COPILOT_MINTED_KEY


def test_copilot_access_token_github_rejects_fails_the_request_without_a_device_login(scene: _Scene) -> None:
    dl.write_copilot_access(scene.rig.copilot_dir, dl.COPILOT_REJECTED_ACCESS)
    served: Final = _serve(scene.call(_Cell("copilot", "chat", False, "httpx")))
    assert served.status == 400, served.text
    assert "Failed to refresh API key" in _error_message(served.text), served.text
    _failure_logged(served)
    reached: Final = scene.rig.peers.auth.drain()
    assert {(request.method, request.target) for request in reached} == {("GET", "/copilot_internal/v2/token")}, (
        _describe(reached)
    )
    assert _describe(scene.rig.peers.api.drain()) == ()


def test_expired_copilot_api_key_with_no_access_token_is_refused_before_any_auth_host_call(scene: _Scene) -> None:
    dl.write_copilot_key(scene.rig.copilot_dir, dl.COPILOT_STORED_KEY, scene.rig.peers.api.url, -3600)
    served: Final = _serve(scene.call(_Cell("copilot", "responses", False, "httpx")))
    assert served.status == 400, served.text
    assert dl.COPILOT_REFUSAL in _error_message(served.text), served.text
    assert scene.rig.peers.auth_connections() == ()
    assert _describe(scene.rig.peers.api.drain()) == ()


_UNUSABLE_CHATGPT_FILES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "invalid-json": "{not json",
        "json-list": "[]",
        "empty-file": "",
        "empty-object": "{}",
        "access-token-int": json.dumps({"access_token": 12345}),
        "access-token-list": json.dumps({"access_token": ["a", "b"]}),
        "access-token-empty": json.dumps({"access_token": ""}),
        "access-token-5kb": json.dumps({"access_token": "x" * 5120}),
        "opaque-token-without-expiry": json.dumps({"access_token": "opaque-token"}),
    }
)


@pytest.mark.parametrize("name", tuple(_UNUSABLE_CHATGPT_FILES))
def test_unusable_chatgpt_auth_file_is_refused_like_a_missing_one_each_time(scene: _Scene, name: str) -> None:
    (scene.rig.chatgpt_dir / "auth.json").write_text(_UNUSABLE_CHATGPT_FILES[name])
    first: Final = _serve(scene.call(_Cell("chatgpt", "responses", False, "httpx")))
    second: Final = _serve(scene.call(_Cell("chatgpt", "responses", False, "httpx")))
    for served in (first, second):
        assert served.status == 400, served.text
        assert dl.CHATGPT_REFUSAL in _error_message(served.text), served.text
    assert scene.rig.peers.auth_connections() == ()
    assert _describe(scene.rig.peers.api.drain()) == ()
    assert (scene.rig.chatgpt_dir / "auth.json").read_text() == _UNUSABLE_CHATGPT_FILES[name]


def test_chatgpt_token_with_a_text_expiry_falls_back_to_the_expiry_inside_the_token(scene: _Scene) -> None:
    dl.write_chatgpt(scene.rig.chatgpt_dir, dl.chatgpt_record(dl.CHATGPT_STORED, expires_at="soon"))
    call: Final = scene.call(_Cell("chatgpt", "responses", False, "httpx"))
    served: Final = _serve(call)
    assert served.status == 200, served.text
    (forwarded,) = _carrying(scene.rig.peers.api.drain(), call.marker)
    assert dl.bearer(forwarded) == dl.CHATGPT_STORED
    assert scene.rig.peers.auth_connections() == ()


_PROVIDER_RESOLUTION_ERROR: Final = "GetLLMProvider Exception"


@dataclass(frozen=True, slots=True)
class _UnusableCopilotFiles:
    files: Mapping[str, str]
    answer: str


_UNUSABLE_COPILOT_FILES: Final[Mapping[str, _UnusableCopilotFiles]] = MappingProxyType(
    {
        "empty-access-token": _UnusableCopilotFiles(MappingProxyType({"access-token": ""}), dl.COPILOT_REFUSAL),
        "blank-access-token": _UnusableCopilotFiles(MappingProxyType({"access-token": "  \n"}), dl.COPILOT_REFUSAL),
        "invalid-api-key-json": _UnusableCopilotFiles(
            MappingProxyType({"api-key.json": "{not json"}), dl.COPILOT_REFUSAL
        ),
        "api-key-list": _UnusableCopilotFiles(MappingProxyType({"api-key.json": "[]"}), _PROVIDER_RESOLUTION_ERROR),
        "api-key-expiry-text": _UnusableCopilotFiles(
            MappingProxyType({"api-key.json": json.dumps({"token": "copilot-text-expiry", "expires_at": "soon"})}),
            _PROVIDER_RESOLUTION_ERROR,
        ),
    }
)


@pytest.mark.parametrize("name", tuple(_UNUSABLE_COPILOT_FILES))
def test_unusable_copilot_token_files_are_refused_like_missing_ones_each_time(scene: _Scene, name: str) -> None:
    variant: Final = _UNUSABLE_COPILOT_FILES[name]
    for file_name, content in variant.files.items():
        (scene.rig.copilot_dir / file_name).write_text(content)
    first: Final = _serve(scene.call(_Cell("copilot", "chat", False, "httpx")))
    second: Final = _serve(scene.call(_Cell("copilot", "chat", False, "httpx")))
    for served in (first, second):
        assert served.status == 400, served.text
        assert variant.answer in _error_message(served.text), served.text
        assert dl.COPILOT_USER_CODE not in served.text
    assert scene.rig.peers.auth_connections() == ()
    assert _describe(scene.rig.peers.api.drain()) == ()


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_a_token_mounted_after_a_refusal_serves_without_a_restart_and_removing_it_refuses_again(
    scene: _Scene, provider: Provider
) -> None:
    cell: Final = _Cell(provider, "chat", False, "httpx")
    before: Final = _serve(scene.call(cell))
    assert before.status == 400, before.text
    scene.store_valid_token(provider)
    mounted: Final = scene.call(cell)
    served: Final = _serve(mounted)
    assert served.status == 200, served.text
    assert set(rv.MARKER.findall(served.text)) == {mounted.marker}, served.text
    dl.clear_tokens(scene.rig.chatgpt_dir, scene.rig.copilot_dir)
    after: Final = _serve(scene.call(cell))
    assert after.status == 400, after.text
    assert _REFUSALS[provider] in _error_message(after.text), after.text
    assert scene.rig.peers.auth_connections() == ()


_COOLDOWN_SECONDS_LEFT: Final = 45


def test_a_recent_device_code_request_does_not_hold_the_worker_for_the_rest_of_the_cooldown(scene: _Scene) -> None:
    dl.write_chatgpt(scene.rig.chatgpt_dir, {"device_code_requested_at": time.time() - (300 - _COOLDOWN_SECONDS_LEFT)})
    started: Final = time.monotonic()
    served: Final = _serve(scene.call(_Cell("chatgpt", "chat", False, "httpx")))
    elapsed: Final = time.monotonic() - started
    assert served.status == 400, served.text
    assert dl.CHATGPT_REFUSAL in _error_message(served.text), served.text
    assert elapsed < _COOLDOWN_SECONDS_LEFT - 15, elapsed
    assert scene.rig.peers.auth_connections() == ()


_NOT_LIVE: Final = re.compile(r"saved to the database, but the model id\(s\) \['([0-9a-f-]+)'\] are not live")


def test_model_created_without_a_token_is_saved_not_live_and_serves_once_the_token_is_mounted(scene: _Scene) -> None:
    gateway: Final = scene.rig.owned.gateway
    name: Final = f"device-login-db-{uuid.uuid4().hex}"
    identity: Final = str(uuid.uuid4())
    created: Final = gateway.request(
        "POST",
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {"model": "chatgpt/gpt-5.6-terra"},
            "model_info": {"id": identity},
        },
    )
    try:
        assert created.status_code == 500, created.text
        not_live: Final = _NOT_LIVE.search(created.text)
        assert not_live is not None and not_live.group(1) == identity, created.text
        assert scene.rig.peers.auth_connections() == ()
        scene.store_valid_token("chatgpt")
        call: Final = _Call(
            scene.rig.base_url(), scene.key, name, "responses", False, "httpx", uuid.uuid4().hex, SimpleQueue()
        )
        served: Final = eventually(lambda: _serve(call), lambda answer: answer.status == 200, seconds=90)
        assert set(rv.MARKER.findall(served.text)) == {call.marker}, served.text
        assert scene.rig.peers.auth_connections() == ()
    finally:
        gateway.request("POST", "/model/delete", {"id": identity})


async def _send(client: httpx.AsyncClient, call: _Call) -> _Served:
    async with client.stream(
        "POST",
        _path(call.endpoint),
        json=_raw_body(call),
        headers={"Authorization": f"Bearer {call.key}", "anthropic-version": "2023-06-01"},
    ) as response:
        raw: Final = await response.aread()
    return _record(call, _Served(response.status_code, raw.decode(), response.headers.get(_CALL_ID, "")))


async def _burst(calls: tuple[_Call, ...]) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=calls[0].base_url, timeout=_CLIENT_SECONDS, trust_env=False) as client:
        return tuple(await asyncio.gather(*(_send(client, call) for call in calls)))


def _chat_response_id(call: _Call, served: _Served) -> str:
    if not call.stream:
        return str(rv.JSON_OBJECT.validate_json(served.text)["id"])
    first: Final = next(line for line in served.text.splitlines() if line.startswith("data: {"))
    return str(rv.JSON_OBJECT.validate_json(first[6:])["id"])


_HTTPX_CELLS: Final = tuple(cell for cell in _CELLS if cell.client == "httpx")


def test_a_burst_of_refusals_leaves_other_models_served_and_logs_every_request_once(scene: _Scene) -> None:
    refused_cells: Final = tuple(_HTTPX_CELLS[index % len(_HTTPX_CELLS)] for index in range(30))
    refused_calls: Final = tuple(scene.call(cell) for cell in refused_cells)
    control_calls: Final = tuple(scene.control(stream=index % 2 == 1) for index in range(10))
    answers: Final = asyncio.run(_burst((*refused_calls, *control_calls)))
    refused: Final = answers[:30]
    controls: Final = answers[30:]
    for cell, answer in zip(refused_cells, refused, strict=True):
        assert answer.status == 400, answer.text
        assert _REFUSALS[cell.provider] in _error_message(answer.text), answer.text
    for call, answer in zip(control_calls, controls, strict=True):
        assert answer.status == 200, answer.text
        assert set(rv.MARKER.findall(answer.text)) == {call.marker}, answer.text
    assert scene.rig.peers.auth_connections() == ()
    received: Final = scene.rig.peers.api.drain()
    assert sorted(rv.newest_marker(request.body.decode()) or "" for request in received) == sorted(
        call.marker for call in control_calls
    ), _describe(received)
    assert {dl.bearer(request) for request in received} == {dl.CONTROL_KEY}
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
            (hashlib.sha256(scene.key.encode()).hexdigest(),),
        ),
        lambda found: len(found) >= 40,
        seconds=70,
    )
    by_status: Final = {str(row["request_id"]): str(row["status"]) for row in rows}
    assert len(by_status) == len(rows) == 40, rows
    for answer in refused:
        assert by_status.get(answer.call_id) == "failure", (answer.call_id, rows)
    for call, answer in zip(control_calls, controls, strict=True):
        (logged,) = [
            request_id for request_id in by_status if rv.same_response(request_id, _chat_response_id(call, answer))
        ]
        assert by_status[logged] == "success", rows


def test_auth_host_outage_refuses_each_refresh_and_the_first_request_after_it_refreshes_once(scene: _Scene) -> None:
    dl.write_chatgpt(scene.rig.chatgpt_dir, dl.chatgpt_record(dl.CHATGPT_EXPIRED, refresh_token=dl.GOOD_REFRESH))
    chatgpt_cells: Final = tuple(cell for cell in _HTTPX_CELLS if cell.provider == "chatgpt")
    scene.rig.peers.switches.refuse_connect.set()
    during: Final = asyncio.run(_burst(tuple(scene.call(chatgpt_cells[index % 6]) for index in range(12))))
    for answer in during:
        assert answer.status == 400, answer.text
        assert dl.CHATGPT_REFUSAL in _error_message(answer.text), answer.text
    for answer in during:
        _failure_logged(answer)
    assert set(scene.rig.peers.auth_connections()) == {f"{dl.CHATGPT_AUTH_HOST}:443"}
    assert _describe(scene.rig.peers.auth.drain()) == ()
    assert _describe(scene.rig.peers.api.drain()) == ()
    scene.rig.peers.switches.refuse_connect.clear()
    first: Final = scene.call(chatgpt_cells[0])
    recovered: Final = _serve(first)
    assert recovered.status == 200, recovered.text
    (refresh,) = scene.rig.peers.auth.drain()
    assert (refresh.method, refresh.target) == ("POST", "/oauth/token")
    after_calls: Final = tuple(scene.call(chatgpt_cells[index % 6]) for index in range(12))
    after: Final = asyncio.run(_burst(after_calls))
    for call, answer in zip(after_calls, after, strict=True):
        assert answer.status == 200, answer.text
        assert set(rv.MARKER.findall(answer.text)) == {call.marker}, answer.text
    assert _describe(scene.rig.peers.auth.drain()) == ()
    forwarded: Final = scene.rig.peers.api.drain()
    assert {dl.bearer(request) for request in forwarded} == {dl.CHATGPT_REFRESHED}, _describe(forwarded)


def _refresh_through_a_stalled_auth_host(scene: _Scene, stall: threading.Event) -> tuple[dl.Hangup, _Served, _Call]:
    dl.write_chatgpt(scene.rig.chatgpt_dir, dl.chatgpt_record(dl.CHATGPT_EXPIRED, refresh_token=dl.GOOD_REFRESH))
    call: Final = scene.call(_Cell("chatgpt", "responses", False, "httpx"))
    stall.set()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending: Final = pool.submit(_httpx, call, 240)
        try:
            dropped: Final = eventually(scene.rig.peers.dropped, lambda hangups: len(hangups) >= 1, seconds=70)
        finally:
            scene.rig.peers.switches.refuse_connect.set()
            stall.clear()
        served: Final = pending.result(timeout=240)
    assert served.status == 400, served.text
    _failure_logged(served)
    scene.rig.peers.switches.refuse_connect.clear()
    assert set(scene.rig.peers.auth_connections()) <= {f"{dl.CHATGPT_AUTH_HOST}:443"}
    return dropped[0], served, call


def _assert_refused_then_refreshed(scene: _Scene, served: _Served) -> None:
    assert served.status == 400, served.text
    assert dl.CHATGPT_REFUSAL in _error_message(served.text), served.text
    assert scene.rig.peers.dropped() == ()
    assert _serve(scene.control(stream=False)).status == 200
    recovered: Final = scene.call(_Cell("chatgpt", "responses", False, "httpx"))
    answer: Final = _serve(recovered)
    assert answer.status == 200, answer.text
    assert set(rv.MARKER.findall(answer.text)) == {recovered.marker}, answer.text
    refreshes: Final = [(request.method, request.target) for request in scene.rig.peers.auth.drain()]
    assert refreshes == [("POST", "/oauth/token")], refreshes
    (forwarded,) = _carrying(scene.rig.peers.api.drain(), recovered.marker)
    assert dl.bearer(forwarded) == dl.CHATGPT_REFRESHED


def test_refresh_call_gives_up_on_a_silent_auth_host_after_30_seconds_and_the_worker_recovers(scene: _Scene) -> None:
    dropped, served, _ = _refresh_through_a_stalled_auth_host(scene, scene.rig.peers.switches.hold_connect)
    assert dropped.authority == f"{dl.CHATGPT_AUTH_HOST}:443"
    assert 25 <= dropped.seconds <= 45, dropped
    _assert_refused_then_refreshed(scene, served)


def test_refresh_call_gives_up_on_a_stalled_tls_handshake_at_the_connect_timeout(scene: _Scene) -> None:
    dropped, served, _ = _refresh_through_a_stalled_auth_host(scene, scene.rig.peers.switches.stall_tls)
    assert dropped.authority == f"{dl.CHATGPT_AUTH_HOST}:443"
    assert 3 <= dropped.seconds <= 20, dropped
    _assert_refused_then_refreshed(scene, served)
