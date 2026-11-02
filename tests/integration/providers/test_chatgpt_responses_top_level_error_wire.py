import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final

import httpx
import pytest
from integration._support import codex_vendor as cv
from integration._support.client import Gateway, gateway_from_environment, object_value, string_value
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.responses_stream import (
    RATE_LIMIT_MESSAGE,
    RATE_LIMIT_PREFIX,
    data_frames,
    error_body,
    openai_error_stream,
)
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_MODEL: Final = "chatgpt/gpt-6.1-sol"
_TOP_LEVEL_RATE_LIMIT: Final = "codex-top-level-rate-limit"
_VENDOR: Final = cv.CodexVendor()


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    wire: Wire


def _respond(request: Request) -> Reply:
    if _TOP_LEVEL_RATE_LIMIT.encode() not in request.body:
        return _VENDOR.respond(request)
    stream: Final = openai_error_stream("resp_" + uuid.uuid4().hex, "rate_limit_exceeded", RATE_LIMIT_MESSAGE)
    return Reply(content_type="text/event-stream", chunks=stream)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("chatgpt-top-level-error-rig")
    with gateway_from_environment() as gateway, wire_server(_respond) as wire:
        overrides: Final = {"CHATGPT_TOKEN_DIR": str(cv.login(directory)), "CHATGPT_API_BASE": wire.url}
        config: Final = cv.proxy_config(directory, probe=False)
        with owned_proxy_process(gateway, directory, overrides, config=config, workers=2) as owned:
            yield _Rig(owned.gateway, wire)


def _rate_limited(rig: _Rig, *, stream: bool) -> httpx.Response:
    rig.wire.drain()
    marker: Final = uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=rig.wire.url, api_key=None)
        body: Final[Mapping[str, JsonValue]] = {
            "model": model,
            "input": [{"role": "user", "content": f"Say marker-{marker} {_TOP_LEVEL_RATE_LIMIT}"}],
            "stream": stream,
            "cache": {"no-cache": True},
        }
        response: Final = rig.gateway.request("POST", "/v1/responses", body)
    (request,) = rig.wire.drain()
    cv.forwarded(request, marker)
    return response


def test_chatgpt_stream_top_level_rate_limit_fails_with_the_provider_code_and_message(rig: _Rig) -> None:
    response: Final = _rate_limited(rig, stream=True)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream"), response.headers
    frames: Final = data_frames(response.text)
    assert [frame["type"] for frame in frames] == ["response.created", "response.failed"], response.text
    error: Final = object_value(object_value(frames[-1]["response"])["error"])
    assert error == {"code": "rate_limit_exceeded", "message": RATE_LIMIT_MESSAGE}, error


def test_chatgpt_aggregated_top_level_rate_limit_answers_429_with_the_provider_message(rig: _Rig) -> None:
    response: Final = _rate_limited(rig, stream=False)
    assert response.status_code == 429, (response.status_code, response.text)
    assert response.headers["content-type"].startswith("application/json"), response.headers
    message: Final = string_value(error_body(response)["message"])
    assert message.startswith(RATE_LIMIT_PREFIX) and RATE_LIMIT_MESSAGE in message, message
    assert "event:" not in message, message
    assert rig.gateway.request("GET", "/health/liveliness", None).status_code == 200
