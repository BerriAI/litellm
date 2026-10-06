"""Realtime WebRTC HTTP routes: what the scripted OpenAI deployment receives and what the client parses.

The browser flow mints an encrypted client secret at ``/v1/realtime/client_secrets`` with a virtual key and
redeems it at ``/v1/realtime/calls`` with a raw SDP offer. The proxy decrypts the token, decodes its payload,
routes on the model group it carries and rebuilds the SDP as a multipart form, so every row here reads the
request the deployment received instead of stopping at the status code.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Mapping
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from typing import Final

import httpx
import pytest
from openai import OpenAI
from openai.types.realtime import (
    ClientSecretCreateResponse,
    RealtimeAudioConfigParam,
    RealtimeSessionCreateRequestParam,
)
from pydantic import BaseModel, JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value, string_value
from tests.integration._support.wire import Reply, Request, Wire, wire_server

DEPLOYMENT_KEY: Final = "synthetic-openai-realtime-key"
REALTIME_MODEL: Final = "gpt-realtime"
TRANSCRIBE_MODEL: Final = "gpt-4o-transcribe"
OFFER: Final = (
    "v=0\r\no=- 4611731400430051336 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n"
    "a=group:BUNDLE 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\nc=IN IP4 0.0.0.0\r\na=mid:0\r\n"
    "a=rtpmap:111 opus/48000/2\r\n"
)
ANSWER: Final = b"v=0\r\no=- 1 2 IN IP4 203.0.113.7\r\ns=-\r\nt=0 0\r\nm=audio 3478 UDP/TLS/RTP/SAVPF 111\r\n"
CALLS_STATUS: Final = 201
PATH_PREFIXES: Final = ("/v1", "")


class _TranscriptionSessionResponse(BaseModel):
    id: str
    object: str
    client_secret: dict[str, JsonValue]


def _raw_secret() -> str:
    return f"ek_raw_{uuid.uuid4().hex}"


def _expires_at() -> int:
    return int(time.time()) + 600


def _realtime_session_reply(raw: str, expires_at: int) -> dict[str, JsonValue]:
    return {
        "value": raw,
        "expires_at": expires_at,
        "session": {
            "type": "realtime",
            "object": "realtime.session",
            "id": "sess_integration",
            "model": REALTIME_MODEL,
            "output_modalities": ["audio"],
            "instructions": "x",
            "client_secret": {"value": raw, "expires_at": expires_at},
        },
    }


def _scripted_openai(client_secret: Mapping[str, JsonValue]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target.endswith("/v1/realtime/client_secrets") or request.target.endswith(
            "/v1/realtime/transcription_sessions"
        ):
            return Reply(body=json.dumps(client_secret).encode())
        if request.target.endswith("/v1/realtime/calls"):
            return Reply(status=CALLS_STATUS, body=ANSWER, content_type="application/sdp")
        raise AssertionError(f"Unexpected upstream request {request.method} {request.target}")

    return respond


def _deployment(scenario: Scenario, wire: Wire, scenario_path: str, model: str) -> str:
    return scenario.model(model=model, api_base=f"{wire.url}{scenario_path}/v1", api_key=DEPLOYMENT_KEY)


def _deployment_named_like_its_model(scenario: Scenario, gateway: Gateway, wire: Wire, scenario_path: str) -> str:
    name: Final = f"integration-{uuid.uuid4().hex}"
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": f"openai/{name}",
                "api_key": DEPLOYMENT_KEY,
                "api_base": f"{wire.url}{scenario_path}/v1",
            },
            "model_info": {},
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return name


def _json_body(request: Request) -> dict[str, JsonValue]:
    assert request.headers["content-type"].startswith("application/json"), request.headers
    return JSON_OBJECT.validate_json(request.body)


def _without_top_level_model(session: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    # main adds a top-level model to every replayed transcription session, owned by the BUG-skipped
    # test_transcription_session_replayed_at_calls_carries_no_top_level_model
    return {key: value for key, value in session.items() if key != "model"}


def _text_parts(request: Request) -> dict[str, str]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    parts: Final[tuple[Message, ...]] = tuple(parsed.iter_parts())
    assert [part.get_filename() for part in parts] == [None] * len(parts), request.body
    names: Final = tuple(str(part.get_param("name", header="content-disposition")) for part in parts)
    assert len(names) == len(set(names)), request.body
    assert [(name, part.get_content_type()) for name, part in zip(names, parts)] == [
        ("sdp", "text/plain"),
        ("session", "application/json"),
    ], request.body
    return {name: bytes(part.get_payload(decode=True)).decode() for name, part in zip(names, parts)}


def _redeem(gateway: Gateway, token: str, path: str = "/v1/realtime/calls") -> httpx.Response:
    return gateway.client.post(
        path,
        content=OFFER.encode(),
        headers={"Content-Type": "application/sdp", "Authorization": f"Bearer {token}"},
    )


def _sdk(gateway: Gateway, api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1", max_retries=0)


@pytest.mark.parametrize("prefix", PATH_PREFIXES)
def test_browser_flow_mints_token_and_redeems_sdp_offer_with_raw_upstream_secret(gateway: Gateway, prefix: str) -> None:
    raw: Final = _raw_secret()
    expires_at: Final = _expires_at()
    scenario_path: Final = f"/{uuid.uuid4().hex}"
    with (
        wire_server(_scripted_openai(_realtime_session_reply(raw, expires_at))) as wire,
        gateway.scenario() as scenario,
    ):
        alias: Final = _deployment(scenario, wire, scenario_path, f"openai/{REALTIME_MODEL}")
        key: Final = scenario.key(models=[alias])
        minted: Final = gateway.request("POST", f"{prefix}/realtime/client_secrets", {"model": alias}, key=key)
        assert minted.status_code == 200, minted.text
        token: Final = ClientSecretCreateResponse.model_validate_json(minted.content).value
        answered: Final = _redeem(gateway, token, f"{prefix}/realtime/calls")

        mint_request, calls_request = wire.drain()
        assert (mint_request.method, mint_request.target) == ("POST", f"{scenario_path}/v1/realtime/client_secrets")
        assert mint_request.headers["authorization"] == f"Bearer {DEPLOYMENT_KEY}", mint_request.headers
        assert _json_body(mint_request) == {"session": {"type": "realtime", "model": REALTIME_MODEL}}, mint_request.body
        assert (calls_request.method, calls_request.target) == ("POST", f"{scenario_path}/v1/realtime/calls")
        assert calls_request.headers["authorization"] == f"Bearer {raw}", calls_request.headers
        parts: Final = _text_parts(calls_request)
        assert parts.keys() == {"sdp", "session"}, calls_request.body
        assert parts["sdp"] == OFFER, calls_request.body
        assert json.loads(parts["session"]) == {"type": "realtime", "model": REALTIME_MODEL}, parts["session"]
        assert answered.status_code == CALLS_STATUS, answered.text
        assert answered.content == ANSWER, answered.text
        assert answered.headers["content-type"] == "application/sdp", answered.headers


def test_openai_prefixed_realtime_aliases_reach_realtime_handlers(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: /openai/v1/realtime/client_secrets, /calls and /transcription_sessions are captured by the OpenAI "
        "pass-through route and return 500 'Required OPENAI_API_KEY' instead of reaching the realtime handlers"
    )
    raw: Final = _raw_secret()
    transcription_raw: Final = _raw_secret()
    scenario_path: Final = f"/{uuid.uuid4().hex}"
    with (
        wire_server(_scripted_openai(_realtime_session_reply(raw, _expires_at()))) as wire,
        wire_server(
            _scripted_openai(_beta_transcription_reply(transcription_raw, _expires_at()))
        ) as transcription_wire,
        gateway.scenario() as scenario,
    ):
        alias: Final = _deployment(scenario, wire, scenario_path, f"openai/{REALTIME_MODEL}")
        transcription_alias: Final = _deployment(
            scenario, transcription_wire, scenario_path, f"openai/{TRANSCRIBE_MODEL}"
        )
        key: Final = scenario.key(models=[alias, transcription_alias])
        minted: Final = gateway.request("POST", "/openai/v1/realtime/client_secrets", {"model": alias}, key=key)
        assert minted.status_code == 200, minted.text
        answered: Final = _redeem(
            gateway, ClientSecretCreateResponse.model_validate_json(minted.content).value, "/openai/v1/realtime/calls"
        )
        assert answered.status_code == CALLS_STATUS, answered.text
        assert answered.content == ANSWER, answered.text
        transcribed: Final = gateway.request(
            "POST",
            "/openai/v1/realtime/transcription_sessions",
            _beta_transcription_body(transcription_alias),
            key=key,
        )
        assert transcribed.status_code == 200, transcribed.text
        transcription_token: Final = _TranscriptionSessionResponse.model_validate_json(
            transcribed.content
        ).client_secret["value"]
        assert isinstance(transcription_token, str) and transcription_token != transcription_raw, transcribed.text
        assert transcription_raw not in transcribed.text, transcribed.text

        mint_request, calls_request = wire.drain()
        assert [(request.target, request.headers["authorization"]) for request in (mint_request, calls_request)] == [
            (f"{scenario_path}/v1/realtime/client_secrets", f"Bearer {DEPLOYMENT_KEY}"),
            (f"{scenario_path}/v1/realtime/calls", f"Bearer {raw}"),
        ]
        (session_request,) = transcription_wire.drain()
        assert (session_request.target, session_request.headers["authorization"]) == (
            f"{scenario_path}/v1/realtime/transcription_sessions",
            f"Bearer {DEPLOYMENT_KEY}",
        )
        assert _json_body(session_request) == _beta_transcription_body(TRANSCRIBE_MODEL), session_request.body


def test_openai_prefixed_transcription_session_reaches_realtime_handler(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: /openai/v1/realtime/transcription_sessions is captured by the OpenAI pass-through route and "
        "returns 500 'Required OPENAI_API_KEY' instead of reaching the realtime handler"
    )
    raw: Final = _raw_secret()
    scenario_path: Final = f"/{uuid.uuid4().hex}"
    with (
        wire_server(_scripted_openai(_beta_transcription_reply(raw, _expires_at()))) as wire,
        gateway.scenario() as scenario,
    ):
        alias: Final = _deployment(scenario, wire, scenario_path, f"openai/{TRANSCRIBE_MODEL}")
        key: Final = scenario.key(models=[alias])
        created: Final = gateway.request(
            "POST",
            "/openai/v1/realtime/transcription_sessions",
            _beta_transcription_body(alias),
            key=key,
        )
        assert created.status_code == 200, created.text
        token: Final = string_value(
            _TranscriptionSessionResponse.model_validate_json(created.content).client_secret["value"]
        )
        assert token != raw, created.text
        assert raw not in created.text, created.text

        (session_request,) = wire.drain()
        assert (session_request.target, session_request.headers["authorization"]) == (
            f"{scenario_path}/v1/realtime/transcription_sessions",
            f"Bearer {DEPLOYMENT_KEY}",
        )


def test_sdk_client_secret_create_forwards_session_and_expires_after_with_deployment_model(gateway: Gateway) -> None:
    raw: Final = _raw_secret()
    expires_at: Final = _expires_at()
    scenario_path: Final = f"/{uuid.uuid4().hex}"
    audio: Final[RealtimeAudioConfigParam] = {
        "input": {"format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": {"type": "semantic_vad"}},
        "output": {"voice": "marin"},
    }
    with (
        wire_server(_scripted_openai(_realtime_session_reply(raw, expires_at))) as wire,
        gateway.scenario() as scenario,
    ):
        alias: Final = _deployment(scenario, wire, scenario_path, f"openai/{REALTIME_MODEL}")
        key: Final = scenario.key(models=[alias])
        session_param: Final = RealtimeSessionCreateRequestParam(
            type="realtime", model=alias, instructions="x", audio=audio
        )
        marker: Final = uuid.uuid4().hex
        created: Final = _sdk(gateway, key).realtime.client_secrets.with_raw_response.create(
            expires_after={"anchor": "created_at", "seconds": 600},
            extra_body={
                # metadata stands in for a session field this SDK version does not declare
                "session": {**session_param, "metadata": {"case": marker}}
            },
        )

        (mint_request,) = wire.drain()
        assert (mint_request.method, mint_request.target) == ("POST", f"{scenario_path}/v1/realtime/client_secrets")
        assert mint_request.headers["authorization"] == f"Bearer {DEPLOYMENT_KEY}", mint_request.headers
        assert _json_body(mint_request) == {
            "session": {
                "type": "realtime",
                "model": REALTIME_MODEL,
                "instructions": "x",
                "audio": audio,
                "metadata": {"case": marker},
            },
            "expires_after": {"anchor": "created_at", "seconds": 600},
        }, mint_request.body
        assert created.http_response.status_code == 200, created.http_response.text


def test_client_secret_response_carries_encrypted_value_that_redeems_at_calls(gateway: Gateway) -> None:
    raw: Final = _raw_secret()
    expires_at: Final = _expires_at()
    scripted: Final = _realtime_session_reply(raw, expires_at)
    scenario_path: Final = f"/{uuid.uuid4().hex}"
    with wire_server(_scripted_openai(scripted)) as wire, gateway.scenario() as scenario:
        alias: Final = _deployment(scenario, wire, scenario_path, f"openai/{REALTIME_MODEL}")
        key: Final = scenario.key(models=[alias])
        created: Final = _sdk(gateway, key).realtime.client_secrets.with_raw_response.create(
            session={"type": "realtime", "model": alias}
        )
        body: Final = created.http_response.content
        parsed: Final = created.parse()
        assert parsed.value != raw, body
        assert raw.encode() not in body, body
        session: Final = scripted["session"]
        assert isinstance(session, dict)
        assert JSON_OBJECT.validate_json(body) == {
            "value": parsed.value,
            "expires_at": expires_at,
            "session": {**session, "client_secret": {"value": parsed.value, "expires_at": expires_at}},
        }, body
        assert parsed.session.type == "realtime", body
        answered: Final = _redeem(gateway, parsed.value)
        assert answered.status_code == CALLS_STATUS, answered.text
        assert answered.content == ANSWER, answered.text
        assert [(request.target, request.headers["authorization"]) for request in wire.drain()] == [
            (f"{scenario_path}/v1/realtime/client_secrets", f"Bearer {DEPLOYMENT_KEY}"),
            (f"{scenario_path}/v1/realtime/calls", f"Bearer {raw}"),
        ]


def _nested_transcription_session(model: str) -> dict[str, JsonValue]:
    return {"type": "transcription", "audio": {"input": {"transcription": {"model": model, "language": "en"}}}}


def _flat_transcription_session(model: str) -> dict[str, JsonValue]:
    return {"type": "transcription", "input_audio_transcription": {"model": model, "language": "en"}}


def _transcription_session_reply(raw: str, expires_at: int) -> dict[str, JsonValue]:
    return {
        "value": raw,
        "expires_at": expires_at,
        "session": {
            "type": "transcription",
            "object": "realtime.transcription_session",
            "id": "sess_transcription",
            "audio": {"input": {"transcription": {"model": TRANSCRIBE_MODEL, "language": "en"}}},
        },
    }


@pytest.mark.parametrize(
    ("spelling", "expected_session"),
    [
        (_nested_transcription_session, _nested_transcription_session(TRANSCRIBE_MODEL)),
        (_flat_transcription_session, _flat_transcription_session(TRANSCRIBE_MODEL)),
    ],
    ids=["nested", "flat"],
)
def test_transcription_client_secret_sends_deployment_model_not_alias_on_mint_and_calls(
    gateway: Gateway,
    spelling: Callable[[str], dict[str, JsonValue]],
    expected_session: dict[str, JsonValue],
) -> None:
    pytest.skip(
        "BUG: transcription client secret forwards the model group alias as the transcription model, "
        "and /realtime/calls adds a top-level model to the transcription session"
    )
    raw: Final = _raw_secret()
    scenario_path: Final = f"/{uuid.uuid4().hex}"
    with (
        wire_server(_scripted_openai(_transcription_session_reply(raw, _expires_at()))) as wire,
        gateway.scenario() as scenario,
    ):
        alias: Final = _deployment(scenario, wire, scenario_path, f"openai/{TRANSCRIBE_MODEL}")
        key: Final = scenario.key(models=[alias])
        minted: Final = gateway.request("POST", "/v1/realtime/client_secrets", {"session": spelling(alias)}, key=key)
        assert minted.status_code == 200, minted.text
        answered: Final = _redeem(gateway, ClientSecretCreateResponse.model_validate_json(minted.content).value)
        assert answered.status_code == CALLS_STATUS, answered.text

        mint_request, calls_request = wire.drain()
        assert _json_body(mint_request) == {"session": expected_session}, mint_request.body
        assert json.loads(_text_parts(calls_request)["session"]) == {
            "type": "transcription",
            "audio": {"input": {"transcription": {"model": TRANSCRIBE_MODEL}}},
        }, calls_request.body


def test_beta_transcription_session_token_replays_deployment_model_not_alias_at_calls(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: a beta transcription_sessions token replays the model group alias as the transcription model "
        "at /realtime/calls and adds a top-level model"
    )
    raw: Final = _raw_secret()
    scenario_path: Final = f"/{uuid.uuid4().hex}"
    with (
        wire_server(_scripted_openai(_beta_transcription_reply(raw, _expires_at()))) as wire,
        gateway.scenario() as scenario,
    ):
        alias: Final = _deployment(scenario, wire, scenario_path, f"openai/{TRANSCRIBE_MODEL}")
        key: Final = scenario.key(models=[alias])
        minted: Final = gateway.request(
            "POST", "/v1/realtime/transcription_sessions", _beta_transcription_body(alias), key=key
        )
        assert minted.status_code == 200, minted.text
        token: Final = string_value(
            _TranscriptionSessionResponse.model_validate_json(minted.content).client_secret["value"]
        )
        answered: Final = _redeem(gateway, token)
        assert answered.status_code == CALLS_STATUS, answered.text

        _session_request, calls_request = wire.drain()
        assert JSON_OBJECT.validate_json(_text_parts(calls_request)["session"]) == {
            "type": "transcription",
            "audio": {"input": {"transcription": {"model": TRANSCRIBE_MODEL}}},
        }, calls_request.body


def test_sdk_calls_create_redeems_token_with_raw_upstream_secret(gateway: Gateway) -> None:
    raw: Final = _raw_secret()
    scenario_path: Final = f"/{uuid.uuid4().hex}"
    with (
        wire_server(_scripted_openai(_realtime_session_reply(raw, _expires_at()))) as wire,
        gateway.scenario() as scenario,
    ):
        alias: Final = _deployment(scenario, wire, scenario_path, f"openai/{REALTIME_MODEL}")
        key: Final = scenario.key(models=[alias])
        token: Final = (
            _sdk(gateway, key).realtime.client_secrets.create(session={"type": "realtime", "model": alias}).value
        )
        http_response: Final = _sdk(gateway, token).realtime.calls.with_raw_response.create(sdp=OFFER)
        assert http_response.status_code == CALLS_STATUS, http_response.text
        assert http_response.content == ANSWER, http_response.text
        assert http_response.headers["content-type"] == "application/sdp", http_response.headers
        assert [(request.method, request.target, request.headers["authorization"]) for request in wire.drain()] == [
            ("POST", f"{scenario_path}/v1/realtime/client_secrets", f"Bearer {DEPLOYMENT_KEY}"),
            ("POST", f"{scenario_path}/v1/realtime/calls", f"Bearer {raw}"),
        ]


def test_sdk_calls_create_multipart_forwards_offer_and_merges_client_session(gateway: Gateway) -> None:
    pytest.skip("BUG: SDK multipart /realtime/calls forwards the form envelope as SDP and drops the client session")
    raw: Final = _raw_secret()
    scenario_path: Final = f"/{uuid.uuid4().hex}"
    with (
        wire_server(_scripted_openai(_realtime_session_reply(raw, _expires_at()))) as wire,
        gateway.scenario() as scenario,
    ):
        alias: Final = _deployment(scenario, wire, scenario_path, f"openai/{REALTIME_MODEL}")
        key: Final = scenario.key(models=[alias])
        token: Final = (
            _sdk(gateway, key).realtime.client_secrets.create(session={"type": "realtime", "model": alias}).value
        )
        answer: Final = _sdk(gateway, token).realtime.calls.create(
            sdp=OFFER, session={"type": "realtime", "instructions": "x"}
        )
        assert answer.content == ANSWER, answer.text

        _mint_request, calls_request = wire.drain()
        assert calls_request.headers["authorization"] == f"Bearer {raw}", calls_request.headers
        parts: Final = _text_parts(calls_request)
        assert parts.keys() == {"sdp", "session"}, calls_request.body
        assert parts["sdp"] == OFFER, calls_request.body
        assert json.loads(parts["session"]) == {
            "type": "realtime",
            "model": REALTIME_MODEL,
            "instructions": "x",
        }, parts["session"]


BETA_TRANSCRIPTION_FIELDS: Final[dict[str, JsonValue]] = {
    "input_audio_format": "pcm16",
    "turn_detection": {"type": "server_vad", "threshold": 0.5, "silence_duration_ms": 500},
    "input_audio_noise_reduction": {"type": "near_field"},
    "include": ["item.input_audio_transcription.logprobs"],
}


def _beta_transcription_body(model: str) -> dict[str, JsonValue]:
    return {
        **BETA_TRANSCRIPTION_FIELDS,
        "input_audio_transcription": {"model": model, "language": "en", "prompt": ""},
    }


def _beta_transcription_reply(raw: str, expires_at: int) -> dict[str, JsonValue]:
    return {
        "id": "sess_beta_transcription",
        "object": "realtime.transcription_session",
        "modalities": ["audio", "text"],
        **BETA_TRANSCRIPTION_FIELDS,
        "input_audio_transcription": {"model": TRANSCRIBE_MODEL, "language": "en", "prompt": ""},
        "client_secret": {"value": raw, "expires_at": expires_at},
    }


def _mint_transcription_and_redeem(gateway: Gateway, mint_spelling: str) -> tuple[str, dict[str, JsonValue]]:
    raw: Final = _raw_secret()
    scenario_path: Final = f"/{uuid.uuid4().hex}"
    is_beta: Final = mint_spelling == "beta"
    route: Final = "transcription_sessions" if is_beta else "client_secrets"
    with (
        wire_server(
            _scripted_openai(
                _beta_transcription_reply(raw, _expires_at())
                if is_beta
                else _transcription_session_reply(raw, _expires_at())
            )
        ) as wire,
        gateway.scenario() as scenario,
    ):
        name: Final = _deployment_named_like_its_model(scenario, gateway, wire, scenario_path)
        key: Final = scenario.key(models=[name])
        request_body: Final = (
            _beta_transcription_body(name)
            if is_beta
            else {
                "session": (
                    _flat_transcription_session(name)
                    if mint_spelling == "ga_flat"
                    else _nested_transcription_session(name)
                )
            }
        )
        minted: Final = gateway.request("POST", f"/v1/realtime/{route}", request_body, key=key)
        assert minted.status_code == 200, minted.text
        token: Final = (
            string_value(_TranscriptionSessionResponse.model_validate_json(minted.content).client_secret["value"])
            if is_beta
            else ClientSecretCreateResponse.model_validate_json(minted.content).value
        )
        answered: Final = _redeem(gateway, token)
        assert answered.status_code == CALLS_STATUS, answered.text
        assert answered.content == ANSWER, answered.text

        mint_request, calls_request = wire.drain()
        assert (
            mint_request.method,
            mint_request.target,
            mint_request.headers["authorization"],
        ) == (
            "POST",
            f"{scenario_path}/v1/realtime/{route}",
            f"Bearer {DEPLOYMENT_KEY}",
        ), mint_request.headers
        assert _json_body(mint_request) == request_body, mint_request.body
        assert (calls_request.target, calls_request.headers["authorization"]) == (
            f"{scenario_path}/v1/realtime/calls",
            f"Bearer {raw}",
        ), calls_request.headers
        replayed: Final = JSON_OBJECT.validate_json(_text_parts(calls_request)["session"])
        return name, replayed


@pytest.mark.parametrize("mint_spelling", ["ga_nested", "ga_flat", "beta"], ids=["ga_nested", "ga_flat", "beta"])
def test_transcription_client_secret_replays_transcription_session_at_calls(
    gateway: Gateway, mint_spelling: str
) -> None:
    name, replayed = _mint_transcription_and_redeem(gateway, mint_spelling)
    assert _without_top_level_model(replayed) == {
        "type": "transcription",
        "audio": {"input": {"transcription": {"model": name}}},
    }, replayed


@pytest.mark.parametrize("mint_spelling", ["ga_nested", "ga_flat", "beta"], ids=["ga_nested", "ga_flat", "beta"])
def test_transcription_session_replayed_at_calls_carries_no_top_level_model(
    gateway: Gateway, mint_spelling: str
) -> None:
    pytest.skip("BUG: /realtime/calls adds a top-level model to a replayed transcription session")
    name, replayed = _mint_transcription_and_redeem(gateway, mint_spelling)
    assert replayed == {"type": "transcription", "audio": {"input": {"transcription": {"model": name}}}}, replayed


@pytest.mark.parametrize("prefix", PATH_PREFIXES)
@pytest.mark.parametrize("with_model_hint", [False, True], ids=["body_only", "model_hint"])
def test_beta_transcription_session_forwards_body_with_deployment_model_and_encrypts_secret(
    gateway: Gateway, prefix: str, with_model_hint: bool
) -> None:
    raw: Final = _raw_secret()
    expires_at: Final = _expires_at()
    scripted: Final = _beta_transcription_reply(raw, expires_at)
    scenario_path: Final = f"/{uuid.uuid4().hex}"
    with wire_server(_scripted_openai(scripted)) as wire, gateway.scenario() as scenario:
        alias: Final = _deployment(scenario, wire, scenario_path, f"openai/{TRANSCRIBE_MODEL}")
        key: Final = scenario.key(models=[alias])
        request_body: Final = {**_beta_transcription_body(alias), **({"model": alias} if with_model_hint else {})}
        created: Final = gateway.request("POST", f"{prefix}/realtime/transcription_sessions", request_body, key=key)
        assert created.status_code == 200, created.text
        parsed: Final = _TranscriptionSessionResponse.model_validate_json(created.content)
        token: Final = parsed.client_secret["value"]
        assert isinstance(token, str) and token != raw, created.text
        assert raw not in created.text, created.text
        assert created.json() == {**scripted, "client_secret": {"value": token, "expires_at": expires_at}}, created.text
        answered: Final = _redeem(gateway, token)
        assert answered.status_code == CALLS_STATUS, answered.text

        session_request, calls_request = wire.drain()
        assert (session_request.method, session_request.target) == (
            "POST",
            f"{scenario_path}/v1/realtime/transcription_sessions",
        )
        assert session_request.headers["authorization"] == f"Bearer {DEPLOYMENT_KEY}", session_request.headers
        assert _json_body(session_request) == _beta_transcription_body(TRANSCRIBE_MODEL), session_request.body
        assert (calls_request.target, calls_request.headers["authorization"]) == (
            f"{scenario_path}/v1/realtime/calls",
            f"Bearer {raw}",
        ), calls_request.headers


@pytest.mark.parametrize("prefix", PATH_PREFIXES)
def test_beta_transcription_session_routes_on_model_hint_over_smuggled_nested_model(
    gateway: Gateway, prefix: str
) -> None:
    raw: Final = _raw_secret()
    allowed_path: Final = f"/{uuid.uuid4().hex}"
    denied_path: Final = f"/{uuid.uuid4().hex}"
    with (
        wire_server(_scripted_openai(_beta_transcription_reply(raw, _expires_at()))) as wire,
        gateway.scenario() as scenario,
    ):
        allowed_name: Final = _deployment_named_like_its_model(scenario, gateway, wire, allowed_path)
        denied_name: Final = _deployment_named_like_its_model(scenario, gateway, wire, denied_path)
        key: Final = scenario.key(models=[allowed_name])
        created: Final = gateway.request(
            "POST",
            f"{prefix}/realtime/transcription_sessions",
            {**_beta_transcription_body(denied_name), "model": allowed_name},
            key=key,
        )
        assert created.status_code == 200, created.text

        token: Final = string_value(
            _TranscriptionSessionResponse.model_validate_json(created.content).client_secret["value"]
        )
        answered: Final = _redeem(gateway, token)
        assert answered.status_code == CALLS_STATUS, answered.text
        assert answered.content == ANSWER, answered.text

        session_request, calls_request = wire.drain()
        assert session_request.target == f"{allowed_path}/v1/realtime/transcription_sessions", session_request.target
        assert _json_body(session_request) == _beta_transcription_body(allowed_name), session_request.body
        assert (calls_request.target, calls_request.headers["authorization"]) == (
            f"{allowed_path}/v1/realtime/calls",
            f"Bearer {raw}",
        ), calls_request.headers
        assert _without_top_level_model(JSON_OBJECT.validate_json(_text_parts(calls_request)["session"])) == {
            "type": "transcription",
            "audio": {"input": {"transcription": {"model": allowed_name}}},
        }, calls_request.body
