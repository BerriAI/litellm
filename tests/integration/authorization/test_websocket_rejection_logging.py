import asyncio
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Final

import psutil
import pytest
import websockets
from integration._support.client import eventually, gateway_from_environment
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from pydantic import JsonValue
from websockets.exceptions import InvalidStatus

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

WEBSOCKET_ROUTES: Final = (
    "/v1/responses",
    "/responses",
    "/v1/realtime",
    "/openai/v1/realtime",
    "/realtime",
    "/openai/v1/responses",
    "/openai_passthrough/v1/responses",
    "/deepgram/v1/listen",
    "/deepgram/listen",
    "/vertex_ai/live",
)
HTTP_ONLY_ROUTES: Final = ("/v1/traces", "/v1/logs", "/v1/chat/completions", "/nope")
HANDSHAKE_SHAPES: Final = (
    "no_header",
    "empty_bearer",
    "basic_scheme",
    "lowercase_bearer",
    "unknown_key",
    "huge_key",
    "duplicate_header",
    "api_key_header",
    "subprotocol_key",
    "malformed_key",
)
BURST_SHAPES: Final = ("no_header", "unknown_key", "denied_model")
CRASH_MARKERS: Final = (
    "Exception in ASGI application",
    "AttributeError: 'WebSocket' object has no attribute 'method'",
    "ERROR: user_api_key_auth.py",
)
OTLP_LENS_MESSAGE: Final = "Send traces and logs directly to the Lens endpoint shown in Lens setup."
REALTIME_TRANSCRIPTION_MODEL: Final = "gpt-realtime-whisper"

Headers = tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class Keys:
    restricted: str
    allowed_model: str


@dataclass(frozen=True, slots=True)
class Refusal:
    status: int
    window: str


@pytest.fixture(scope="module")
def owned(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    with (
        gateway_from_environment() as base,
        owned_proxy_process(
            base,
            tmp_path_factory.mktemp("websocket-rejection"),
            {"LITELLM_DISABLE_NO_REDIS_WARNING": "true"},
            workers=2,
        ) as proxy,
    ):
        yield proxy
    stopped_log: Final = proxy.log.read_text(errors="replace")
    assert _crash_lines(stopped_log) == (), _marker_lines(stopped_log)


@pytest.fixture(scope="module")
def keys(owned: OwnedProxy) -> Iterator[Keys]:
    with owned.gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        yield Keys(restricted=scenario.key(models=[allowed]), allowed_model=allowed)


def _tag(label: str) -> str:
    return f"integration-{label}-{uuid.uuid4().hex[:12]}"


def _ws_url(owned: OwnedProxy, path: str, query: str) -> str:
    base: Final = str(owned.gateway.client.base_url).rstrip("/").replace("http://", "ws://")
    return f"{base}{path}?{query}"


def _log_after(owned: OwnedProxy, offset: int) -> str:
    return owned.log.read_bytes()[offset:].decode(errors="replace")


def _crash_lines(window: str) -> tuple[str, ...]:
    return tuple(marker for marker in CRASH_MARKERS if marker in window)


def _marker_lines(text: str) -> tuple[str, ...]:
    return tuple(line for line in text.splitlines() if any(marker in line for marker in CRASH_MARKERS))


def _shapes(keys: Keys) -> Mapping[str, Headers]:
    unknown: Final = f"sk-{uuid.uuid4().hex}"
    return MappingProxyType(
        {
            "no_header": (),
            "empty_bearer": (("Authorization", "Bearer "),),
            "basic_scheme": (("Authorization", "Basic aW50ZWdyYXRpb246eA=="),),
            "lowercase_bearer": (("Authorization", f"bearer {unknown}"),),
            "unknown_key": (("Authorization", f"Bearer {unknown}"),),
            "huge_key": (("Authorization", f"Bearer sk-{'a' * 5120}"),),
            "duplicate_header": (("Authorization", f"Bearer {unknown}"), ("Authorization", f"Bearer {unknown}")),
            "api_key_header": (("api-key", unknown),),
            "subprotocol_key": (("Sec-WebSocket-Protocol", f"openai-insecure-api-key.{unknown}"),),
            "malformed_key": (("Authorization", "Bearer integration-not-a-virtual-key"),),
            "denied_model": (("Authorization", f"Bearer {keys.restricted}"),),
        }
    )


async def _refused_status(owned: OwnedProxy, path: str, headers: Headers, query: str) -> int:
    with pytest.raises(InvalidStatus) as refused:
        async with websockets.connect(_ws_url(owned, path, query), additional_headers=headers):
            pass
    return refused.value.response.status_code


async def _refused(owned: OwnedProxy, path: str, headers: Headers, query: str) -> Refusal:
    offset: Final = owned.log.stat().st_size
    status: Final = await _refused_status(owned, path, headers, query)
    window: Final = eventually(
        lambda: _log_after(owned, offset),
        lambda text: f'"WebSocket {path}?{query}" 403' in text,
        seconds=30,
    )
    return Refusal(status, window)


@pytest.mark.parametrize("shape", HANDSHAKE_SHAPES)
@pytest.mark.parametrize("path", WEBSOCKET_ROUTES)
async def test_a_refused_handshake_answers_403_and_logs_no_asgi_crash(
    owned: OwnedProxy, keys: Keys, path: str, shape: str
) -> None:
    refusal: Final = await _refused(owned, path, _shapes(keys)[shape], f"model={_tag('ws')}")
    assert refusal.status == 403, refusal.window
    assert _crash_lines(refusal.window) == (), refusal.window


@pytest.mark.parametrize("path", WEBSOCKET_ROUTES)
async def test_a_denied_model_handshake_logs_one_warning_line(owned: OwnedProxy, keys: Keys, path: str) -> None:
    tag: Final = _tag("denied")
    refusal: Final = await _refused(owned, path, _shapes(keys)["denied_model"], f"model={tag}")
    assert refusal.status == 403, refusal.window
    assert refusal.window.count(f"Tried to access {tag}") == 1, refusal.window
    assert _crash_lines(refusal.window) == (), refusal.window


async def test_a_realtime_denial_on_the_resolved_transcription_model_logs_one_warning_line(
    owned: OwnedProxy, keys: Keys
) -> None:
    tag: Final = _tag("transcription")
    refusal: Final = await _refused(
        owned, "/v1/realtime", _shapes(keys)["denied_model"], f"intent=transcription&tag={tag}"
    )
    assert refusal.status == 403, refusal.window
    assert refusal.window.count(f"Tried to access {REALTIME_TRANSCRIPTION_MODEL}") == 1, refusal.window
    assert _crash_lines(refusal.window) == (), refusal.window


async def test_a_realtime_handshake_without_a_model_is_refused_without_a_denial_line(
    owned: OwnedProxy, keys: Keys
) -> None:
    refusal: Final = await _refused(owned, "/v1/realtime", _shapes(keys)["denied_model"], f"tag={_tag('nomodel')}")
    assert refusal.status == 403, refusal.window
    assert "Tried to access" not in refusal.window, refusal.window
    assert _crash_lines(refusal.window) == (), refusal.window


@pytest.mark.parametrize("path", HTTP_ONLY_ROUTES)
async def test_a_websocket_handshake_on_an_http_only_route_is_refused_with_403(owned: OwnedProxy, path: str) -> None:
    refusal: Final = await _refused(owned, path, (), f"model={_tag('httponly')}")
    assert refusal.status == 403, refusal.window
    assert _crash_lines(refusal.window) == (), refusal.window


def _http_window(owned: OwnedProxy, offset: int, access_line: str) -> str:
    return eventually(lambda: _log_after(owned, offset), lambda text: access_line in text, seconds=30)


@pytest.mark.parametrize("content_type", ("application/json", "application/x-protobuf"))
@pytest.mark.parametrize("path", ("/v1/traces", "/v1/logs"))
def test_an_otlp_ingest_post_still_answers_410_in_the_caller_s_encoding(
    owned: OwnedProxy, path: str, content_type: str
) -> None:
    tag: Final = _tag("otlp")
    offset: Final = owned.log.stat().st_size
    response: Final = owned.gateway.client.post(
        f"{path}?tag={tag}", content=b"", headers={"content-type": content_type}
    )
    assert response.status_code == 410, response.text
    assert response.headers["content-type"].split(";", 1)[0] == content_type, response.headers
    assert OTLP_LENS_MESSAGE.encode() in response.content, response.content
    window: Final = _http_window(owned, offset, f'"POST {path}?tag={tag} HTTP/1.1" 410')
    assert _crash_lines(window) == (), window


def test_a_get_on_the_otlp_path_is_not_otlp_ingest_and_answers_a_plain_401(owned: OwnedProxy) -> None:
    tag: Final = _tag("otlpget")
    offset: Final = owned.log.stat().st_size
    response: Final = owned.gateway.client.get(
        f"/v1/traces?tag={tag}", headers={"content-type": "application/x-protobuf"}
    )
    assert response.status_code == 401, response.text
    assert response.headers["content-type"].split(";", 1)[0] == "application/json", response.headers
    assert response.json()["error"]["code"] == "401", response.text
    window: Final = _http_window(owned, offset, f'"GET /v1/traces?tag={tag} HTTP/1.1" 401')
    assert _crash_lines(window) == (), window


def test_an_unknown_http_route_answers_a_plain_404(owned: OwnedProxy) -> None:
    tag: Final = _tag("nope")
    offset: Final = owned.log.stat().st_size
    response: Final = owned.gateway.request("POST", f"/nope?tag={tag}", {})
    assert response.status_code == 404, response.text
    assert response.json() == {"detail": "Not Found"}, response.text
    window: Final = _http_window(owned, offset, f'"POST /nope?tag={tag} HTTP/1.1" 404')
    assert _crash_lines(window) == (), window


def test_a_bad_request_on_a_plain_route_keeps_its_detail(owned: OwnedProxy, keys: Keys) -> None:
    tag: Final = _tag("tokens")
    offset: Final = owned.log.stat().st_size
    response: Final = owned.gateway.request("POST", f"/utils/token_counter?tag={tag}", {"model": keys.allowed_model})
    assert response.status_code == 400, response.text
    assert response.json() == {"detail": "prompt or messages or contents must be provided"}, response.text
    window: Final = _http_window(owned, offset, f'"POST /utils/token_counter?tag={tag} HTTP/1.1" 400')
    assert _crash_lines(window) == (), window


def _denied_body(path: str, model: str) -> Mapping[str, JsonValue]:
    message: Final[dict[str, JsonValue]] = {"role": "user", "content": "integration denial"}
    match path:
        case "/v1/messages":
            return {"model": model, "max_tokens": 4, "messages": [message]}
        case "/v1/responses":
            return {"model": model, "input": "integration denial"}
        case _:
            return {"model": model, "messages": [message]}


@pytest.mark.parametrize("path", ("/v1/chat/completions", "/v1/messages", "/v1/responses"))
def test_an_http_model_denial_logs_one_warning_line_and_no_traceback(owned: OwnedProxy, keys: Keys, path: str) -> None:
    tag: Final = _tag("httpdenied")
    offset: Final = owned.log.stat().st_size
    response: Final = owned.gateway.request("POST", path, _denied_body(path, tag), key=keys.restricted)
    assert response.status_code == 403, response.text
    window: Final = eventually(
        lambda: _log_after(owned, offset), lambda text: f"Tried to access {tag}" in text, seconds=30
    )
    assert window.count(f"Tried to access {tag}") == 1, window
    assert _crash_lines(window) == (), window


def _workers(owned: OwnedProxy) -> tuple[psutil.Process, ...]:
    return tuple(child for child in psutil.Process(owned.process.pid).children() if _is_worker(child))


def _is_worker(child: psutil.Process) -> bool:
    try:
        return "spawn_main" in " ".join(child.cmdline()) and child.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


async def _burst(owned: OwnedProxy, shapes: Mapping[str, Headers], label: str) -> str:
    plan: Final = tuple((path, shape, _tag(label)) for path, shape in product(WEBSOCKET_ROUTES, BURST_SHAPES))
    offset: Final = owned.log.stat().st_size
    statuses: Final = await asyncio.gather(
        *(_refused_status(owned, path, shapes[shape], f"model={tag}") for path, shape, tag in plan)
    )
    assert tuple(statuses) == (403,) * len(plan), statuses
    access_lines: Final = tuple(f'"WebSocket {path}?model={tag}" 403' for path, _, tag in plan)
    window: Final = eventually(
        lambda: _log_after(owned, offset),
        lambda text: all(line in text for line in access_lines),
        seconds=60,
    )
    assert window.count(f"Tried to access integration-{label}-") == len(WEBSOCKET_ROUTES), window
    return window


async def test_a_rejection_burst_survives_a_killed_worker_without_an_asgi_crash(owned: OwnedProxy, keys: Keys) -> None:
    shapes: Final = _shapes(keys)
    before: Final = await _burst(owned, shapes, "burst")
    assert _crash_lines(before) == (), before
    victim: Final = eventually(lambda: _workers(owned), lambda workers: len(workers) == 2)[0]
    victim.kill()
    during: Final = await _refused(owned, "/v1/responses", shapes["unknown_key"], f"model={_tag('during')}")
    assert during.status == 403, during.window
    assert _crash_lines(during.window) == (), during.window
    eventually(
        lambda: frozenset(worker.pid for worker in _workers(owned)),
        lambda pids: len(pids) == 2 and victim.pid not in pids,
        seconds=graceful_stop_seconds(),
    )
    after: Final = await _burst(owned, shapes, "respawned")
    assert _crash_lines(after) == (), after
