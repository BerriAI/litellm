"""Access control on realtime client secrets.

The mint at ``/v1/realtime/client_secrets`` is the only point where the virtual key is checked, because the
encrypted token it returns is the whole credential at ``/v1/realtime/calls``. Every refusal here asserts the
scripted OpenAI deployment saw nothing, which is what keeps a refused caller off the proxy's provider account.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from hashlib import sha256
from typing import Final

import httpx
import pytest
from openai.types.realtime import ClientSecretCreateResponse
from pydantic import BaseModel, JsonValue

from tests.integration._support.client import Gateway, Scenario, object_value
from tests.integration._support.wire import Reply, Request, Wire, wire_server

DEPLOYMENT_KEY: Final = "synthetic-openai-realtime-key"
OFFER: Final = "v=0\r\no=- 1 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
ANSWER: Final = b"v=0\r\no=- 3 4 IN IP4 203.0.113.7\r\ns=-\r\nt=0 0\r\n"


class _OpenAIError(BaseModel):
    message: str
    type: str
    param: str | None
    code: str


class _OpenAIErrorEnvelope(BaseModel):
    error: _OpenAIError


def _client_secret(raw: str, expires_at: int) -> dict[str, JsonValue]:
    return {
        "value": raw,
        "expires_at": expires_at,
        "session": {
            "type": "realtime",
            "object": "realtime.session",
            "id": "sess_access",
            "client_secret": {"value": raw, "expires_at": expires_at},
        },
    }


def _scripted_openai(client_secrets: Iterator[Mapping[str, JsonValue]]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target.endswith("/v1/realtime/client_secrets"):
            return Reply(body=json.dumps(next(client_secrets)).encode())
        if request.target.endswith("/v1/realtime/calls"):
            return Reply(status=201, body=ANSWER, content_type="application/sdp")
        raise AssertionError(f"Unexpected upstream request {request.method} {request.target}")

    return respond


def _endless_client_secrets(raw: str) -> Iterator[Mapping[str, JsonValue]]:
    while True:
        yield _client_secret(raw, int(time.time()) + 600)


def _deployment(scenario: Scenario, wire: Wire, model: str) -> str:
    return scenario.model(model=model, api_base=f"{wire.url}/{uuid.uuid4().hex}/v1", api_key=DEPLOYMENT_KEY)


def _mint(gateway: Gateway, body: Mapping[str, JsonValue], key: str) -> httpx.Response:
    return gateway.request("POST", "/v1/realtime/client_secrets", body, key=key)


def _redeem(gateway: Gateway, headers: Mapping[str, str]) -> httpx.Response:
    return gateway.client.post(
        "/v1/realtime/calls", content=OFFER.encode(), headers={"Content-Type": "application/sdp", **headers}
    )


def _key_info(gateway: Gateway, key: str) -> dict[str, JsonValue]:
    return object_value(gateway.get("/key/info", {"key": sha256(key.encode()).hexdigest()})["info"])


def _nested_transcription(model: str) -> dict[str, JsonValue]:
    return {"type": "transcription", "audio": {"input": {"transcription": {"model": model}}}}


@pytest.mark.parametrize(
    "spelling",
    ["session_model", "top_level_model", "nested_transcription_model", "default_model"],
)
def test_client_secret_mint_refuses_model_outside_key_scope_before_reaching_openai(
    gateway: Gateway, spelling: str
) -> None:
    raw: Final = f"ek_raw_{uuid.uuid4().hex}"
    with wire_server(_scripted_openai(_endless_client_secrets(raw))) as wire, gateway.scenario() as scenario:
        rt_allowed: Final = _deployment(scenario, wire, "openai/gpt-realtime")
        whisper_allowed: Final = _deployment(scenario, wire, "openai/gpt-4o-transcribe")
        rt_denied: Final = _deployment(scenario, wire, "openai/gpt-realtime")
        whisper_denied: Final = _deployment(scenario, wire, "openai/gpt-4o-transcribe")
        key: Final = scenario.key(models=[rt_allowed, whisper_allowed])
        denied_bodies: Final[Mapping[str, Mapping[str, JsonValue]]] = {
            "session_model": {"session": {"type": "realtime", "model": rt_denied}},
            "top_level_model": {"model": rt_denied},
            "nested_transcription_model": {"model": whisper_allowed, "session": _nested_transcription(whisper_denied)},
            "default_model": {},
        }

        refused: Final = _mint(gateway, denied_bodies[spelling], key)
        assert refused.status_code == 403, refused.text
        error: Final = _OpenAIErrorEnvelope.model_validate_json(refused.content).error
        assert (error.type, error.param, error.code) == ("key_model_access_denied", "model", "403"), refused.text
        assert wire.drain() == (), refused.text

        realtime_minted: Final = _mint(gateway, {"session": {"type": "realtime", "model": rt_allowed}}, key)
        assert realtime_minted.status_code == 200, realtime_minted.text
        transcription_minted: Final = _mint(gateway, {"session": _nested_transcription(whisper_allowed)}, key)
        assert transcription_minted.status_code == 200, transcription_minted.text
        assert [request.target.rsplit("/", 1)[-1] for request in wire.drain()] == ["client_secrets"] * 2
        info: Final = _key_info(gateway, key)
        assert (info["models"], info["blocked"], info["spend"]) == ([rt_allowed, whisper_allowed], None, 0.0), info


@pytest.mark.parametrize(
    ("refusal", "expected_status", "expected_type"),
    [("blocked", 401, "auth_error"), ("over_budget", 422, "budget_exceeded")],
)
def test_client_secret_mint_refuses_blocked_or_exhausted_key(
    gateway: Gateway, refusal: str, expected_status: int, expected_type: str
) -> None:
    raw: Final = f"ek_raw_{uuid.uuid4().hex}"
    with wire_server(_scripted_openai(_endless_client_secrets(raw))) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "openai/gpt-realtime")
        key: Final = scenario.key(
            models=[model], **({"max_budget": 0.0001, "spend": 1.0} if refusal == "over_budget" else {})
        )
        if refusal == "blocked":
            gateway.post("/key/block", {"key": key})

        # the message pins which enforcement path refused (budget_reservation also refuses with a different message)
        expected_message: Final = (
            "Authentication Error, Key is blocked. Update via `/key/unblock` if you're an admin."
            if refusal == "blocked"
            else f"Budget has been exceeded! Key=key (sk-...{key[-4:]}) Current cost: 1.0, Max budget: 0.0001"
        )
        refused: Final = _mint(gateway, {"model": model}, key)
        assert refused.status_code == expected_status, refused.text
        error: Final = _OpenAIErrorEnvelope.model_validate_json(refused.content).error
        assert (error.type, error.code, error.message) == (
            expected_type,
            str(expected_status),
            expected_message,
        ), refused.text
        assert wire.drain() == (), refused.text
        info: Final = _key_info(gateway, key)
        assert (info["blocked"], info["max_budget"]) == ((True, None) if refusal == "blocked" else (None, 0.0001)), info


def _flip_one_character(token: str) -> str:
    middle: Final = len(token) // 2
    return token[:middle] + ("A" if token[middle] != "A" else "B") + token[middle + 1 :]


def test_calls_refuses_missing_forged_and_proxy_credentials_without_reaching_openai(gateway: Gateway) -> None:
    raw: Final = f"ek_raw_{uuid.uuid4().hex}"
    with wire_server(_scripted_openai(_endless_client_secrets(raw))) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "openai/gpt-realtime")
        key: Final = scenario.key(models=[model])
        minted: Final = _mint(gateway, {"model": model}, key)
        assert minted.status_code == 200, minted.text
        token: Final = ClientSecretCreateResponse.model_validate_json(minted.content).value
        assert wire.drain()[0].target.endswith("/v1/realtime/client_secrets")

        refusals: Final[dict[str, tuple[Mapping[str, str], dict[str, str]]]] = {
            "no_header": ({}, {"error": "Missing or invalid Authorization header"}),
            "virtual_key": ({"Authorization": f"Bearer {key}"}, {"error": "Invalid or expired token"}),
            "master_key": ({"Authorization": f"Bearer {gateway.key}"}, {"error": "Invalid or expired token"}),
            "flipped": (
                {"Authorization": f"Bearer {_flip_one_character(token)}"},
                {"error": "Invalid or expired token"},
            ),
        }
        observed: Final = {
            name: (response.status_code, response.json())
            for name, (headers, _expected) in refusals.items()
            for response in (_redeem(gateway, headers),)
        }
        assert observed == {name: (401, expected) for name, (_headers, expected) in refusals.items()}, observed
        assert wire.drain() == ()

        answered: Final = _redeem(gateway, {"Authorization": f"Bearer {token}"})
        assert (answered.status_code, answered.content) == (201, ANSWER), answered.text
        assert [(request.target.rsplit("/", 1)[-1], request.headers["authorization"]) for request in wire.drain()] == [
            ("calls", f"Bearer {raw}")
        ]


def test_calls_refuses_token_past_upstream_expiry_and_accepts_a_fresh_one(gateway: Gateway) -> None:
    expired_raw: Final = f"ek_expired_{uuid.uuid4().hex}"
    fresh_raw: Final = f"ek_fresh_{uuid.uuid4().hex}"
    client_secrets: Final = iter(
        (_client_secret(expired_raw, int(time.time()) - 1), _client_secret(fresh_raw, int(time.time()) + 600))
    )
    with wire_server(_scripted_openai(client_secrets)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "openai/gpt-realtime")
        key: Final = scenario.key(models=[model])
        expired: Final = _mint(gateway, {"model": model}, key)
        assert expired.status_code == 200, expired.text
        refused: Final = _redeem(
            gateway,
            {"Authorization": f"Bearer {ClientSecretCreateResponse.model_validate_json(expired.content).value}"},
        )
        assert refused.status_code == 401, refused.text
        assert refused.json() == {"error": "Token has expired"}, refused.text

        fresh: Final = _mint(gateway, {"model": model}, key)
        assert fresh.status_code == 200, fresh.text
        answered: Final = _redeem(
            gateway,
            {"Authorization": f"Bearer {ClientSecretCreateResponse.model_validate_json(fresh.content).value}"},
        )
        assert (answered.status_code, answered.content) == (201, ANSWER), answered.text
        assert [(request.target.rsplit("/", 1)[-1], request.headers["authorization"]) for request in wire.drain()] == [
            ("client_secrets", f"Bearer {DEPLOYMENT_KEY}"),
            ("client_secrets", f"Bearer {DEPLOYMENT_KEY}"),
            ("calls", f"Bearer {fresh_raw}"),
        ]
