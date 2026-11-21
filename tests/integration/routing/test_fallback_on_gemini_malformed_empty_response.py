import json
import re
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias
from urllib.parse import unquote

import httpx
import pytest
from integration._support.client import Gateway
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_BACKEND: Final = "gemini-4-carbon"
_API_KEY: Final = "synthetic-gemini-key"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-central1"
_MODEL_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{_BACKEND}"
_BACKUP_ANSWER: Final = "The backup deployment answered."
_MALFORMED: Final = "malformed-empty"
_STOP_EMPTY: Final = "stop-empty"
_TAG: Final = re.compile(r"<scenario ([a-z-]+)> nonce=([0-9a-f]{32})")
_USAGE: Final[Mapping[str, int]] = MappingProxyType(
    {"promptTokenCount": 11, "candidatesTokenCount": 5, "totalTokenCount": 16}
)

Provider: TypeAlias = Literal["gemini", "vertex_ai"]
Endpoint: TypeAlias = Literal["chat", "messages", "responses"]
_PROVIDERS: Final[tuple[Provider, ...]] = ("gemini", "vertex_ai")
_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")
_PATHS: Final = MappingProxyType(
    {"chat": "/v1/chat/completions", "messages": "/v1/messages", "responses": "/v1/responses"}
)


def _gemini_reply(reason: str, parts: tuple[Mapping[str, JsonValue], ...] | None) -> Reply:
    content: Final[dict[str, JsonValue]] = (
        {"role": "model"} if parts is None else {"role": "model", "parts": [dict(part) for part in parts]}
    )
    return Reply(
        body=json.dumps(
            {"candidates": [{"content": content, "finishReason": reason, "index": 0}], "usageMetadata": dict(_USAGE)}
        ).encode()
    )


def _deployment_of(request: Request) -> str:
    return unquote(request.target).lstrip("/").partition("/")[0]


def _respond(request: Request) -> Reply:
    found: Final = _TAG.search(request.body.decode())
    deployment: Final = _deployment_of(request)
    if found is None or ":generateContent" not in request.target:
        return Reply(status=404, body=json.dumps({"error": f"unexpected {request.target}"}).encode())
    if deployment == "backup":
        return _gemini_reply("STOP", ({"text": _BACKUP_ANSWER},))
    return _gemini_reply("MALFORMED_FUNCTION_CALL" if found.group(1) == _MALFORMED else "STOP", None)


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    wire: Wire
    primary: Mapping[Provider, str]
    backup: str


def _vertex_model(gateway: Gateway, api_base: str) -> dict[str, JsonValue]:
    return {
        "model": f"vertex_ai/{_BACKEND}",
        "api_base": f"{api_base}{_MODEL_PATH}",
        "api_key": None,
        "vertex_project": _PROJECT,
        "vertex_location": _LOCATION,
        "vertex_credentials": service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
    }


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway) -> Iterator[_Rig]:
    with wire_server(_respond) as wire, session_gateway.scenario() as scenario:
        primary: Final[Mapping[Provider, str]] = MappingProxyType(
            {
                "gemini": scenario.model(model=f"gemini/{_BACKEND}", api_base=f"{wire.url}/primary", api_key=_API_KEY),
                "vertex_ai": scenario.model(**_vertex_model(session_gateway, f"{wire.url}/primary")),
            }
        )
        backup: Final = scenario.model(model=f"gemini/{_BACKEND}", api_base=f"{wire.url}/backup", api_key=_API_KEY)
        yield _Rig(session_gateway, wire, primary, backup)


def _body(endpoint: Endpoint, model: str, prompt: str) -> dict[str, JsonValue]:
    match endpoint:
        case "chat":
            return {"model": model, "messages": [{"role": "user", "content": prompt}], "num_retries": 0}
        case "messages":
            return {"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": prompt}]}
        case "responses":
            return {"model": model, "input": prompt}


@dataclass(frozen=True, slots=True)
class _Observed:
    response: httpx.Response
    hit: tuple[str, ...]


def _send(rig: _Rig, endpoint: Endpoint, provider: Provider, scenario: str) -> _Observed:
    nonce: Final = uuid.uuid4().hex
    response: Final = rig.gateway.request(
        "POST",
        _PATHS[endpoint],
        {
            **_body(endpoint, rig.primary[provider], f"<scenario {scenario}> nonce={nonce}"),
            "fallbacks": [rig.backup],
        },
    )
    hit: Final = tuple(_deployment_of(request) for request in rig.wire.drain() if nonce in request.body.decode())
    return _Observed(response, hit)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_regression_malformed_empty_primary_answer_falls_back_to_the_backup(
    rig: _Rig, endpoint: Endpoint, provider: Provider
) -> None:
    observed: Final = _send(rig, endpoint, provider, _MALFORMED)
    assert observed.response.status_code == 200, observed.response.text
    assert _BACKUP_ANSWER in observed.response.text, observed.response.text
    assert observed.hit[0] == "primary" and observed.hit[-1] == "backup", observed.hit


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_guard_stop_with_no_parts_on_the_primary_does_not_fall_back(rig: _Rig, provider: Provider) -> None:
    observed: Final = _send(rig, "chat", provider, _STOP_EMPTY)
    assert observed.response.status_code == 200, observed.response.text
    assert observed.hit == ("primary",), observed.hit
