"""A deployment that declares ``custom_llm_provider`` and names a bare model, on every route the declaration reaches.

Each row drives the client a user drives, then reads what the scripted provider received. A declared deployment must
send the same request its ``openai/`` prefixed twin sends, and a value only the caller controls must not change where
the deployment's credentials go.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from email.parser import BytesParser
from email.policy import HTTP
from itertools import product
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import websockets
from openai import OpenAI
from pydantic import JsonValue
from websockets.exceptions import ConnectionClosed

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value, string_value
from tests.integration._support.upstream import delete_scenario, register_scenario
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration._support.ports import allocate_port
from tests.integration.cost_calculation.cost_tracking_case import RealtimeResponse

_KEY: Final = "synthetic-declared-provider-key"
_FORMS: Final = ("declared", "prefixed", "blank")
_IMAGE: Final = "aW1n"
_AUDIO: Final = bytes(range(256)) * 8
_WAV: Final = (
    b"RIFF\x30\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x40\x1f\x00\x00\x40\x1f\x00\x00\x01\x00\x08\x00data\x08\x00\x00\x00"
    b"\x00\x01\x02\x03\x04\x05\x06\x07"
)
_OFFER: Final = (
    "v=0\r\no=- 4611731400430051336 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n"
    "a=group:BUNDLE 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\nc=IN IP4 0.0.0.0\r\na=mid:0\r\n"
    "a=rtpmap:111 opus/48000/2\r\n"
)
_ANSWER: Final = b"v=0\r\no=- 1 2 IN IP4 203.0.113.7\r\ns=-\r\nt=0 0\r\nm=audio 3478 UDP/TLS/RTP/SAVPF 111\r\n"
_CALLS_STATUS: Final = 201
_TRANSCRIPTION_SESSION_FIELDS: Final[dict[str, JsonValue]] = {"input_audio_format": "pcm16"}


def _bare() -> str:
    return f"bare-{uuid.uuid4().hex[:12]}"


def _provider_params(form: str, bare: str) -> dict[str, JsonValue]:
    match form:
        case "declared":
            return {"model": bare, "custom_llm_provider": "openai"}
        case "prefixed":
            return {"model": f"openai/{bare}"}
        case "blank":
            return {"model": f"openai/{bare}", "custom_llm_provider": ""}
        case _:
            raise AssertionError(form)


def _deployment(scenario: Scenario, wire: Wire, form: str, bare: str) -> str:
    return scenario.model(**_provider_params(form, bare), api_base=f"{wire.url}/v1", api_key=_KEY)


def _sdk(gateway: Gateway, key: str) -> OpenAI:
    return OpenAI(
        base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
        api_key=key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=30),
    )


def _json_body(request: Request) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(request.body)


def _form_fields(request: Request) -> dict[str, str]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers
    return {
        str(part.get_param("name", header="content-disposition")): bytes(part.get_payload(decode=True)).decode()
        for part in parsed.iter_parts()
        if part.get_filename() is None
    }


def _image_reply(bare: str, prompt: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/images/generations"), request
        assert request.headers["authorization"] == f"Bearer {_KEY}", request.headers
        assert _json_body(request) == {"model": bare, "prompt": prompt}, request.body
        return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": _IMAGE}]}).encode())

    return respond


def _speech_reply(bare: str, text: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/audio/speech"), request
        assert request.headers["authorization"] == f"Bearer {_KEY}", request.headers
        assert _json_body(request) == {"model": bare, "input": text, "voice": "alloy", "response_format": "mp3"}
        return Reply(body=_AUDIO, content_type="audio/mpeg")

    return respond


def _transcription_reply(bare: str, prompt: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/audio/transcriptions"), request
        assert request.headers["authorization"] == f"Bearer {_KEY}", request.headers
        fields: Final = _form_fields(request)
        assert (fields["model"], fields["prompt"]) == (bare, prompt), fields
        return Reply(body=json.dumps({"text": prompt}).encode())

    return respond


def _targets(wire: Wire) -> list[tuple[str, str]]:
    return [(request.method, request.target) for request in wire.drain()]


@pytest.mark.parametrize("form", _FORMS)
def test_image_generation_reaches_the_provider_for_a_bare_model(gateway: Gateway, form: str) -> None:
    bare: Final = _bare()
    prompt: Final = f"a lighthouse at dawn {bare}"
    with wire_server(_image_reply(bare, prompt)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, form, bare)
        with _sdk(gateway, scenario.key(models=[model])) as sdk:
            result: Final = sdk.images.generate(model=model, prompt=prompt)
        assert [image.b64_json for image in result.data or []] == [_IMAGE], result
        assert _targets(wire) == [("POST", "/v1/images/generations")]


@pytest.mark.parametrize("form", _FORMS)
def test_speech_reaches_the_provider_for_a_bare_model(gateway: Gateway, form: str) -> None:
    bare: Final = _bare()
    text: Final = f"hello {bare}"
    with wire_server(_speech_reply(bare, text)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, form, bare)
        with _sdk(gateway, scenario.key(models=[model])) as sdk:
            response: Final = sdk.audio.speech.with_raw_response.create(
                model=model, input=text, voice="alloy", response_format="mp3"
            )
        assert response.status_code == 200, response.read()
        assert response.content == _AUDIO
        assert _targets(wire) == [("POST", "/v1/audio/speech")]


@pytest.mark.parametrize("form", _FORMS)
def test_transcription_reaches_the_provider_for_a_bare_model(gateway: Gateway, form: str) -> None:
    bare: Final = _bare()
    prompt: Final = f"hello {bare}"
    with wire_server(_transcription_reply(bare, prompt)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, form, bare)
        with _sdk(gateway, scenario.key(models=[model])) as sdk:
            transcript: Final = sdk.audio.transcriptions.create(
                model=model, file=("a.wav", _WAV, "audio/wav"), prompt=prompt
            )
        assert transcript.text == prompt
        assert _targets(wire) == [("POST", "/v1/audio/transcriptions")]


@pytest.mark.parametrize("form", _FORMS)
def test_test_connect_reaches_the_provider_for_a_bare_image_model(gateway: Gateway, form: str) -> None:
    bare: Final = _bare()

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/images/generations"), request
        assert request.headers["authorization"] == f"Bearer {_KEY}", request.headers
        assert _json_body(request)["model"] == bare, request.body
        return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": _IMAGE}]}).encode())

    with wire_server(respond) as wire:
        api_base: Final = f"{wire.url}/v1"
        response: Final = gateway.request(
            "POST",
            "/health/test_connection",
            {
                "litellm_params": {**_provider_params(form, bare), "api_base": api_base, "api_key": _KEY},
                "mode": "image_generation",
            },
        )
        assert response.status_code == 200, response.text
        assert object_value(response.json())["status"] == "success", response.text
        assert _targets(wire) == [("POST", "/v1/images/generations")]


def _ws_url(gateway: Gateway, model: str) -> str:
    base: Final = str(gateway.client.base_url).rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
    return f"{base}/v1/realtime?model={model}"


async def _first_event(url: str, key: str) -> dict[str, JsonValue]:
    try:
        async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {key}"}) as connection:
            return JSON_OBJECT.validate_json(await asyncio.wait_for(connection.recv(), 30))
    except ConnectionClosed as closed:
        return {"type": "closed", "code": 1006 if closed.rcvd is None else closed.rcvd.code}


def _upgrades(upstream_url: str, scenario_id: str) -> list[tuple[JsonValue, JsonValue]]:
    with httpx.Client(base_url=upstream_url, timeout=5, trust_env=False) as upstream:
        observed: Final = JSON_OBJECT.validate_json(upstream.get("/__observations").content)["requests"]
    assert isinstance(observed, list), observed
    return [
        (request["path"], object_value(request["body"]).get("query"))
        for request in map(object_value, observed)
        if request["method"] == "WEBSOCKET" and request["authorization"] == f"Bearer {scenario_id}"
    ]


def _realtime_scenario(scenario: Scenario) -> str:
    scenario_id: Final = f"declared-provider-{uuid.uuid4().hex[:12]}"
    handle: Final = register_scenario(scenario_id, RealtimeResponse(content_type="application/x-realtime", events=()))
    scenario.cleanups.callback(delete_scenario, handle)
    return scenario_id


@pytest.mark.parametrize("form", _FORMS)
def test_realtime_session_opens_at_the_provider_for_a_bare_model(gateway: Gateway, form: str) -> None:
    bare: Final = _bare()
    with gateway.scenario() as scenario:
        scenario_id: Final = _realtime_scenario(scenario)
        model: Final = scenario.model(**_provider_params(form, bare), api_key=scenario_id)
        first: Final = asyncio.run(_first_event(_ws_url(gateway, model), scenario.key(models=[model])))
        assert first["type"] == "session.created", first
        assert _upgrades(gateway.upstream_url, scenario_id) == [("/v1/realtime", [["model", bare]])]


def test_realtime_follows_the_declared_provider_over_a_contradicting_prefix(gateway: Gateway) -> None:
    bare: Final = _bare()
    with gateway.scenario() as scenario:
        scenario_id: Final = _realtime_scenario(scenario)
        model: Final = scenario.model(
            model=f"azure/{bare}", custom_llm_provider="openai", api_key=scenario_id, api_version="2025-04-01-preview"
        )
        first: Final = asyncio.run(_first_event(_ws_url(gateway, model), scenario.key(models=[model])))
        assert first["type"] == "session.created", first
        assert _upgrades(gateway.upstream_url, scenario_id) == [("/v1/realtime", [["model", f"azure/{bare}"]])]


def test_image_generation_follows_the_declared_provider_over_a_contradicting_prefix(gateway: Gateway) -> None:
    bare: Final = _bare()
    with wire_server(_image_reply(f"azure/{bare}", "declared wins")) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"azure/{bare}",
            custom_llm_provider="openai",
            api_base=f"{wire.url}/v1",
            api_key=_KEY,
            api_version="2025-04-01-preview",
        )
        response: Final = gateway.request("POST", "/v1/images/generations", {"model": model, "prompt": "declared wins"})
        assert response.status_code == 200, response.text
        assert _targets(wire) == [("POST", "/v1/images/generations")]


def _client_secret(raw: str) -> dict[str, JsonValue]:
    return {"value": raw, "expires_at": int(time.time()) + 600, "session": {"type": "realtime", "model": "scripted"}}


def _scripted_realtime(raw: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target.endswith("/v1/realtime/client_secrets"):
            return Reply(body=json.dumps(_client_secret(raw)).encode())
        if request.target.endswith("/v1/realtime/transcription_sessions"):
            return Reply(
                body=json.dumps(
                    {
                        "id": "sess_declared",
                        "object": "realtime.transcription_session",
                        "client_secret": {"value": raw, "expires_at": int(time.time()) + 600},
                    }
                ).encode()
            )
        if request.target.endswith("/v1/realtime/calls"):
            return Reply(status=_CALLS_STATUS, body=_ANSWER, content_type="application/sdp")
        raise AssertionError(f"Unexpected upstream request {request.method} {request.target}")

    return respond


def _redeem(gateway: Gateway, token: str) -> httpx.Response:
    return gateway.client.post(
        "/v1/realtime/calls",
        content=_OFFER.encode(),
        headers={"Content-Type": "application/sdp", "Authorization": f"Bearer {token}"},
    )


@pytest.mark.parametrize("form", _FORMS)
def test_browser_call_reaches_the_provider_for_a_bare_realtime_model(gateway: Gateway, form: str) -> None:
    bare: Final = _bare()
    raw: Final = f"ek_raw_{uuid.uuid4().hex}"
    with wire_server(_scripted_realtime(raw)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, form, bare)
        key: Final = scenario.key(models=[model])
        minted: Final = gateway.request("POST", "/v1/realtime/client_secrets", {"model": model}, key=key)
        assert minted.status_code == 200, minted.text
        answered: Final = _redeem(gateway, string_value(object_value(minted.json())["value"]))
        assert (answered.status_code, answered.content) == (_CALLS_STATUS, _ANSWER), answered.text
        mint_request, calls_request = wire.drain()
        assert (mint_request.target, mint_request.headers["authorization"]) == (
            "/v1/realtime/client_secrets",
            f"Bearer {_KEY}",
        )
        assert _json_body(mint_request) == {"session": {"type": "realtime", "model": bare}}, mint_request.body
        assert (calls_request.target, calls_request.headers["authorization"]) == ("/v1/realtime/calls", f"Bearer {raw}")


@pytest.mark.parametrize("form", _FORMS)
def test_transcription_session_reaches_the_provider_for_a_bare_model(gateway: Gateway, form: str) -> None:
    bare: Final = _bare()
    raw: Final = f"ek_raw_{uuid.uuid4().hex}"
    with wire_server(_scripted_realtime(raw)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, form, bare)
        created: Final = gateway.request(
            "POST",
            "/v1/realtime/transcription_sessions",
            {**_TRANSCRIPTION_SESSION_FIELDS, "input_audio_transcription": {"model": model}},
            key=scenario.key(models=[model]),
        )
        assert created.status_code == 200, created.text
        assert object_value(created.json())["id"] == "sess_declared", created.text
        (session_request,) = wire.drain()
        assert (session_request.target, session_request.headers["authorization"]) == (
            "/v1/realtime/transcription_sessions",
            f"Bearer {_KEY}",
        )
        assert _json_body(session_request) == {
            **_TRANSCRIPTION_SESSION_FIELDS,
            "input_audio_transcription": {"model": bare},
        }, session_request.body


_IMAGES: Final = "/v1/images/generations"
_SPEECH: Final = "/v1/audio/speech"
_TRANSCRIPTIONS: Final = "/v1/audio/transcriptions"
_CLIENT_SECRETS: Final = "/v1/realtime/client_secrets"
_NON_STRING_HINTS: Final = MappingProxyType(
    {"int": "5", "list": '["openai"]', "twice": '"openai", "custom_llm_provider": 5'}
)
_SECRET_HINTS: Final = MappingProxyType({"int": "5", "list": '["openai"]', "azure": '"azure"'})


def _unexpected(request: Request) -> Reply:
    raise AssertionError(f"Unexpected upstream request {request.method} {request.target}")


def _fields(route: str, marker: str) -> dict[str, JsonValue]:
    match route:
        case "/v1/images/generations":
            return {"prompt": marker}
        case "/v1/audio/speech":
            return {"input": marker, "voice": "alloy", "response_format": "mp3"}
        case "/v1/realtime/client_secrets":
            return {"session": {"type": "realtime"}}
        case _:
            raise AssertionError(route)


def _post_with_hint(gateway: Gateway, route: str, model: str, marker: str, hint: str) -> httpx.Response:
    body: Final = json.dumps({"model": model, **_fields(route, marker)})
    return gateway.client.post(
        route,
        content=f'{body[:-1]}, "custom_llm_provider": {hint}}}'.encode(),
        headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
    )


@pytest.mark.parametrize("route", (_IMAGES, _SPEECH))
@pytest.mark.parametrize("hint", tuple(_NON_STRING_HINTS))
def test_non_string_provider_hint_from_the_caller_is_refused_before_the_provider(
    gateway: Gateway, route: str, hint: str
) -> None:
    with wire_server(_unexpected) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "prefixed", _bare())
        response: Final = _post_with_hint(gateway, route, model, "refused", _NON_STRING_HINTS[hint])
        assert response.status_code == 400, response.text
        assert "custom_llm_provider" in string_value(object_value(object_value(response.json())["error"])["message"])
        assert _targets(wire) == []


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("hint", tuple(_SECRET_HINTS))
def test_provider_hint_from_the_caller_does_not_move_a_realtime_secret(gateway: Gateway, form: str, hint: str) -> None:
    with wire_server(_scripted_realtime(f"ek_raw_{uuid.uuid4().hex}")) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, form, _bare())
        response: Final = _post_with_hint(gateway, _CLIENT_SECRETS, model, "", _SECRET_HINTS[hint])
        assert response.status_code == 200, response.text
        assert [(request.target, request.headers["authorization"]) for request in wire.drain()] == [
            (_CLIENT_SECRETS, f"Bearer {_KEY}")
        ]


@pytest.mark.parametrize("route", (_IMAGES, _SPEECH))
def test_empty_provider_hint_from_the_caller_keeps_the_deployment_route(gateway: Gateway, route: str) -> None:
    bare: Final = _bare()
    respond: Final = _image_reply(bare, "kept") if route == _IMAGES else _speech_reply(bare, "kept")
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "prefixed", bare)
        response: Final = _post_with_hint(gateway, route, model, "kept", '""')
        assert response.status_code == 200, response.text
        assert _targets(wire) == [("POST", route)]


@pytest.mark.parametrize("route", (_IMAGES, _SPEECH))
def test_oversized_provider_hint_from_the_caller_never_reaches_the_provider(gateway: Gateway, route: str) -> None:
    with wire_server(_unexpected) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "prefixed", _bare())
        _post_with_hint(gateway, route, model, "oversized", json.dumps("x" * 5120))
        assert _targets(wire) == []
        assert gateway.client.get("/health/liveliness").status_code == 200


def _send(client: httpx.AsyncClient, route: str, model: str, marker: str) -> Awaitable[httpx.Response]:
    if route == _TRANSCRIPTIONS:
        return client.post(route, data={"model": model, "prompt": marker}, files={"file": ("a.wav", _WAV, "audio/wav")})
    return client.post(route, json={"model": model, **_fields(route, marker)})


async def _send_all(gateway: Gateway, calls: tuple[tuple[str, str, str], ...]) -> tuple[httpx.Response, ...]:
    async with httpx.AsyncClient(
        base_url=gateway.client.base_url,
        headers={"Authorization": f"Bearer {gateway.key}"},
        timeout=120,
        trust_env=False,
    ) as client:
        return tuple(await asyncio.gather(*(_send(client, route, model, marker) for route, model, marker in calls)))


def _error_message(response: httpx.Response) -> str:
    return string_value(object_value(object_value(response.json())["error"])["message"]).split("\n\n")[0]


def _scripted_outage(request: Request) -> Reply:
    return Reply(status=500, body=json.dumps({"error": {"message": "scripted outage", "code": "500"}}).encode())


@pytest.mark.parametrize(
    ("route", "known_model"), ((_IMAGES, "dall-e-3"), (_SPEECH, "tts-1"), (_TRANSCRIPTIONS, "whisper-1"))
)
def test_declared_azure_deployment_fails_like_its_prefixed_twin(gateway: Gateway, route: str, known_model: str) -> None:
    with wire_server(_scripted_outage) as wire, gateway.scenario() as scenario:
        declared: Final = scenario.model(
            model=known_model,
            custom_llm_provider="azure",
            api_base=wire.url,
            api_key=_KEY,
            api_version="2025-04-01-preview",
            num_retries=2,
        )
        prefixed: Final = scenario.model(
            model=f"azure/{known_model}",
            api_base=wire.url,
            api_key=_KEY,
            api_version="2025-04-01-preview",
            num_retries=2,
        )
        (declared_response,) = asyncio.run(_send_all(gateway, ((route, declared, "outage"),)))
        declared_targets: Final = _targets(wire)
        (prefixed_response,) = asyncio.run(_send_all(gateway, ((route, prefixed, "outage"),)))
        assert declared_response.status_code == 500, declared_response.text
        assert declared_targets, declared_response.text
        assert (declared_response.status_code, _error_message(declared_response), declared_targets) == (
            prefixed_response.status_code,
            _error_message(prefixed_response),
            _targets(wire),
        )


def _burst_reply(request: Request) -> Reply:
    match request.target:
        case "/v1/images/generations":
            return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": _IMAGE}]}).encode())
        case "/v1/audio/speech":
            return Reply(body=_AUDIO, content_type="audio/mpeg")
        case "/v1/audio/transcriptions":
            return Reply(body=json.dumps({"text": _form_fields(request)["prompt"]}).encode())
        case _:
            raise AssertionError(request.target)


def _marker(request: Request) -> str:
    if request.target == _TRANSCRIPTIONS:
        return _form_fields(request)["prompt"]
    body: Final = _json_body(request)
    return string_value(body["prompt"] if request.target == _IMAGES else body["input"])


def test_burst_across_declared_and_dead_deployments_reaches_each_live_provider_once(gateway: Gateway) -> None:
    dead_base: Final = f"http://127.0.0.1:{allocate_port()}/v1"
    with wire_server(_burst_reply) as wire, gateway.scenario() as scenario:
        live: Final = tuple(
            (route, _deployment(scenario, wire, form, _bare()))
            for route, form in product((_IMAGES, _SPEECH, _TRANSCRIPTIONS), _FORMS)
        )
        dead: Final = tuple(
            (route, scenario.model(model=_bare(), custom_llm_provider="openai", api_base=dead_base, api_key=_KEY))
            for route in (_IMAGES, _SPEECH)
        )
        live_calls: Final = tuple(
            (route, model, f"live-{uuid.uuid4().hex}") for (route, model), _ in product(live, range(5))
        )
        dead_calls: Final = tuple(
            (route, model, f"dead-{uuid.uuid4().hex}") for (route, model), _ in product(dead, range(5))
        )
        responses: Final = asyncio.run(_send_all(gateway, live_calls + dead_calls))
        live_responses: Final = responses[: len(live_calls)]
        dead_responses: Final = responses[len(live_calls) :]
        assert [response.status_code for response in live_responses] == [200] * len(live_calls), [
            response.text for response in live_responses if response.status_code != 200
        ]
        assert sorted(_marker(request) for request in wire.drain()) == sorted(marker for _, _, marker in live_calls)
        assert all(response.status_code >= 500 and _error_message(response) for response in dead_responses), [
            (response.status_code, response.text) for response in dead_responses
        ]
        assert gateway.client.get("/health/liveliness").status_code == 200
        assert gateway.chat(scenario.model(), text=f"burst control {uuid.uuid4().hex}")["choices"]
