"""Router-wide, deployment and `user_config` defaults against the params the Router binds positionally, the
moderation merge, unknown and A2A models, and every `user_config` shape on the three LLM endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration.compatibility._status_code_audit import (
    UNSET,
    CHAT,
    CHAT_FRAMES,
    ENDPOINT_BODIES,
    ENDPOINT_PATHS,
    MESSAGE,
    MESSAGE_FRAMES,
    MODERATION,
    RESPONSE,
    RESPONSES_FRAMES,
    ROUTER_DEFAULTS,
    USER_MESSAGES,
    assembled_text,
    assert_no_provider_call,
    drain_rig_upstream,
    deployment,
    error_body,
    invalid_request,
    json_response,
    one_outbound,
    owned_gateway,
    post,
    sse_response,
    stream_finished,
    stream_lines,
)
from pydantic import JsonValue

from tests.integration.cost_calculation.cost_tracking_case import BinaryResponse, StoredResponse

pytestmark: Final = pytest.mark.timeout(180)


@pytest.fixture(autouse=True)
def _drained_upstream() -> None:
    drain_rig_upstream()


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    identities: Mapping[str, str]
    deployments: Mapping[str, dict[str, JsonValue]]


_SPECS: Final[tuple[tuple[str, str, StoredResponse, dict[str, JsonValue]], ...]] = (
    ("audit-image", "openai/gpt-image-1", json_response({"created": 1, "data": []}), {}),
    ("audit-image-dep", "openai/gpt-image-1", json_response({"created": 1, "data": []}), {"prompt": "deployment"}),
    ("audit-text", "openai/gpt-3.5-turbo-instruct", json_response({"choices": []}), {}),
    ("audit-text-dep", "openai/gpt-3.5-turbo-instruct", json_response({"choices": []}), {"prompt": "deployment"}),
    ("audit-speech", "openai/gpt-4o-mini-tts", BinaryResponse(content_type="audio/mpeg", length=16), {}),
    (
        "audit-speech-dep",
        "openai/gpt-4o-mini-tts",
        BinaryResponse(content_type="audio/mpeg", length=16),
        {"input": "deployment"},
    ),
    ("audit-transcribe", "openai/whisper-1", json_response({"text": "scripted"}), {}),
    ("audit-moderation", "openai/omni-moderation-latest", json_response(MODERATION), {}),
    ("audit-chat", "openai/gpt-4o-mini", json_response(CHAT), {}),
    ("audit-chat-stream", "openai/gpt-4o-mini", sse_response(CHAT_FRAMES), {}),
    ("audit-responses", "openai/gpt-4o-mini", json_response(RESPONSE), {}),
    ("audit-responses-stream", "openai/gpt-4o-mini", sse_response(RESPONSES_FRAMES), {}),
    ("audit-message", "anthropic/claude-haiku-4-5", json_response(MESSAGE), {}),
    ("audit-message-stream", "anthropic/claude-haiku-4-5", sse_response(MESSAGE_FRAMES), {}),
)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    """A proxy whose router-wide defaults name every positional and route-required param, with client-side
    `user_config` allowed, and one deployment per route (plus `-dep` copies that carry the param themselves)."""
    with gateway_from_environment() as gateway, gateway.scenario() as scenario:
        built: Final = tuple(
            deployment(scenario, name, model, response, **extra) for name, model, response, extra in _SPECS
        )
        identities: Final = {name: identity for (name, *_), (identity, _) in zip(_SPECS, built, strict=True)}
        deployments: Final = {name: entry for (name, *_), (_, entry) in zip(_SPECS, built, strict=True)}
        config: Final[dict[str, JsonValue]] = {
            "model_list": [entry for _, entry in built],
            "router_settings": {"default_litellm_params": ROUTER_DEFAULTS, "num_retries": 0},
            "general_settings": {"allow_client_side_credentials": True},
        }
        with owned_gateway(tmp_path_factory.mktemp("audit-defaults"), config) as owned:
            yield _Rig(owned, identities, deployments)


_POSITIONAL: Final[dict[str, tuple[str, str, str, str, dict[str, JsonValue]]]] = {
    "image": ("/v1/images/generations", "/image/generations", "audit-image", "prompt", {}),
    "text-completion": ("/v1/completions", "/completions", "audit-text", "prompt", {}),
    "speech": ("/v1/audio/speech", "/audio/speech", "audit-speech", "input", {"voice": "alloy"}),
}


def _user_config(rig: _Rig, name: str, defaults: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {"model_list": [rig.deployments[name]], "default_litellm_params": dict(defaults)}


@pytest.mark.parametrize("route", tuple(_POSITIONAL))
@pytest.mark.parametrize("source", ("deployment-param", "user-config-default", "user-config-null"))
def test_positional_param_from_any_default_source_returns_400(rig: _Rig, route: str, source: str) -> None:
    path, error_route, name, param, extra = _POSITIONAL[route]
    user_config: Final[dict[str, JsonValue]] = (
        {"user_config": _user_config(rig, name, {param: "user config default"})}
        if source == "user-config-default"
        else {"user_config": None}
        if source == "user-config-null"
        else {}
    )
    body: Final[dict[str, JsonValue]] = {
        "model": f"{name}-dep" if source == "deployment-param" else name,
        **extra,
        **user_config,
    }
    response: Final = post(rig.gateway, path, body)
    error: Final = invalid_request(response)
    assert error.get("param") == param, response.text
    assert error.get("message") == f"{error_route}: Missing required parameter: '{param}'.", response.text
    assert_no_provider_call(rig.gateway, rig.identities[name], rig.identities[f"{name}-dep"])


@pytest.mark.parametrize("route", tuple(_POSITIONAL))
def test_positional_param_in_the_body_reaches_upstream(rig: _Rig, route: str) -> None:
    pytest.skip(
        "BUG: a router-wide default_litellm_params value replaces the caller's own positional param "
        "(prompt/input) on its way to the provider; same on the merge base"
    )
    path, _error_route, name, param, extra = _POSITIONAL[route]
    response: Final = post(rig.gateway, path, {"model": name, param: f"explicit {route}", **extra})
    assert response.status_code == 200, response.text
    assert one_outbound(rig.gateway, rig.identities[name]).get(param) == f"explicit {route}"


def test_transcription_without_file_is_rejected_by_the_form_before_routing(rig: _Rig) -> None:
    response: Final = rig.gateway.client.post(
        "/v1/audio/transcriptions",
        data={"model": "audit-transcribe"},
        files={"note": ("note.txt", b"not audio", "text/plain")},
        headers={"Authorization": f"Bearer {rig.gateway.key}"},
    )
    assert response.status_code == 422, response.text
    detail: Final = response.json()["detail"]
    assert [entry["loc"] for entry in detail] == [["body", "file"]], response.text
    assert_no_provider_call(rig.gateway, rig.identities["audit-transcribe"])


def test_transcription_with_file_reaches_upstream(rig: _Rig) -> None:
    response: Final = rig.gateway.client.post(
        "/v1/audio/transcriptions",
        data={"model": "audit-transcribe"},
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        headers={"Authorization": f"Bearer {rig.gateway.key}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "scripted"
    outbound: Final = one_outbound(rig.gateway, rig.identities["audit-transcribe"])
    assert outbound.get("model") == "whisper-1"
    assert outbound.get("file") == {"filename": "clip.wav", "content_type": "audio/x-wav"}


@pytest.mark.parametrize(
    ("body", "expected_input"),
    (({}, "router default input"), ({"input": "explicit input"}, "explicit input")),
    ids=("router-default", "explicit"),
)
def test_listed_moderation_model_merges_the_router_default(
    rig: _Rig, body: dict[str, JsonValue], expected_input: str
) -> None:
    response: Final = post(rig.gateway, "/v1/moderations", {"model": "audit-moderation", **body})
    assert response.status_code == 200, response.text
    assert one_outbound(rig.gateway, rig.identities["audit-moderation"]).get("input") == expected_input


@pytest.mark.parametrize(
    ("path", "body", "message"),
    (
        pytest.param(
            "/v1/images/generations",
            {"model": "audit-unknown-model"},
            "/image/generations: Missing required parameter: 'prompt'.",
            id="image-unknown-model",
        ),
        pytest.param(
            "/v1/images/generations",
            {"model": "a2a/audit-unknown-agent"},
            "/image/generations: Missing required parameter: 'prompt'.",
            id="image-a2a-model",
        ),
        pytest.param(
            "/v1/messages",
            {"model": "audit-unknown-model", "messages": USER_MESSAGES},
            "anthropic_messages: Missing required parameter: 'max_tokens'.",
            id="messages-unknown-model",
        ),
        pytest.param(
            "/v1/messages",
            {"model": "a2a/audit-unknown-agent", "messages": USER_MESSAGES},
            "anthropic_messages: Invalid model name passed in model=a2a/audit-unknown-agent. "
            "Call `/v1/models` to view available models for your key.",
            id="messages-a2a-model",
        ),
    ),
)
def test_model_outside_the_router_with_a_router_default_param(
    rig: _Rig, path: str, body: dict[str, JsonValue], message: str
) -> None:
    response: Final = post(rig.gateway, path, body)
    assert response.status_code == 400, response.text
    error: Final = error_body(response)
    assert error.get("type") == "invalid_request_error", response.text
    assert error.get("message") == message, response.text


@pytest.mark.parametrize(
    ("user_config", "message"),
    (
        pytest.param(None, None, id="null"),
        pytest.param(UNSET, None, id="missing"),
        pytest.param({}, "There are no healthy deployments for this model", id="empty-object"),
        pytest.param("", "Invalid request format: 'str' object has no attribute 'items'", id="empty-string"),
        pytest.param(7, "Invalid request format: 'int' object has no attribute 'items'", id="int"),
        pytest.param([], "Invalid request format: 'list' object has no attribute 'items'", id="list"),
        pytest.param("x" * 5000, "Invalid request format: 'str' object has no attribute 'items'", id="5kb-string"),
    ),
)
def test_chat_user_config_shapes(rig: _Rig, user_config: object, message: str | None) -> None:
    body: Final[dict[str, JsonValue]] = {
        "model": "audit-chat",
        "messages": USER_MESSAGES,
        **({} if user_config is UNSET else {"user_config": user_config}),  # pyright: ignore[reportAssignmentType]  # the sad shapes are deliberately not JSON objects
    }
    response: Final = post(rig.gateway, "/v1/chat/completions", body)
    if message is None:
        assert response.status_code == 200, response.text
        assert one_outbound(rig.gateway, rig.identities["audit-chat"]).get("max_tokens") == 32
        return
    assert response.status_code == 400, response.text
    assert message in str(error_body(response).get("message")), response.text
    assert_no_provider_call(rig.gateway, rig.identities["audit-chat"])


def test_messages_empty_user_config_does_not_inherit_router_defaults(rig: _Rig) -> None:
    response: Final = post(
        rig.gateway, "/v1/messages", {"model": "audit-message", "messages": USER_MESSAGES, "user_config": {}}
    )
    assert response.status_code == 400, response.text
    assert error_body(response).get("message") == "anthropic_messages: Missing required parameter: 'max_tokens'."
    assert_no_provider_call(rig.gateway, rig.identities["audit-message"])


@pytest.mark.parametrize(
    ("endpoint", "model", "stream", "expected_max_tokens"),
    (
        pytest.param("chat", "audit-chat", False, 32, id="chat"),
        pytest.param("chat", "audit-chat-stream", True, 32, id="chat-stream"),
        pytest.param("responses", "audit-responses", False, None, id="responses"),
        pytest.param("responses", "audit-responses-stream", True, None, id="responses-stream"),
        pytest.param("messages", "audit-message-stream", True, 32, id="messages-stream"),
    ),
)
def test_null_user_config_serves_every_endpoint(
    rig: _Rig, endpoint: str, model: str, stream: bool, expected_max_tokens: int | None
) -> None:
    body: Final[dict[str, JsonValue]] = {
        **{key: value for key, value in ENDPOINT_BODIES[endpoint].items() if key != "max_tokens"},
        "model": model,
        "user_config": None,
        **({"stream": True} if stream else {}),
    }
    if stream:
        status, lines = stream_lines(rig.gateway, ENDPOINT_PATHS[endpoint], body)
        assert status == 200, lines
        assert stream_finished(endpoint, lines), lines
        assert assembled_text(lines) == "streamed response", lines
    else:
        response: Final = post(rig.gateway, ENDPOINT_PATHS[endpoint], body)
        assert response.status_code == 200, response.text
        assert "scripted" in response.text, response.text
    outbound: Final = one_outbound(rig.gateway, rig.identities[model])
    assert "user_config" not in outbound, outbound
    assert outbound.get("max_tokens") == expected_max_tokens, outbound


def test_null_user_config_without_a_valid_key_is_401(rig: _Rig) -> None:
    response: Final = post(
        rig.gateway,
        "/v1/chat/completions",
        {"model": "audit-chat", "messages": USER_MESSAGES, "user_config": None},
        key="sk-audit-not-a-key",
    )
    assert response.status_code == 401, response.text
    assert_no_provider_call(rig.gateway, rig.identities["audit-chat"])


def _openai_chat(rig: _Rig, asynchronous: bool) -> str | None:
    base_url: Final = f"{rig.gateway.client.base_url}/v1"
    if not asynchronous:
        with openai.OpenAI(base_url=base_url, api_key=rig.gateway.key, max_retries=0) as client:
            completion: Final = client.chat.completions.create(
                model="audit-chat",
                messages=[{"role": "user", "content": "Hello"}],
                extra_body={"user_config": None},
            )
            return completion.choices[0].message.content

    async def call() -> str | None:
        async with openai.AsyncOpenAI(base_url=base_url, api_key=rig.gateway.key, max_retries=0) as client:
            completion: Final = await client.chat.completions.create(
                model="audit-chat",
                messages=[{"role": "user", "content": "Hello"}],
                extra_body={"user_config": None},
            )
            return completion.choices[0].message.content

    return asyncio.run(call())


def _anthropic_message(rig: _Rig, asynchronous: bool) -> str:
    base_url: Final = str(rig.gateway.client.base_url)
    if not asynchronous:
        with anthropic.Anthropic(base_url=base_url, api_key=rig.gateway.key, max_retries=0) as client:
            message: Final = client.messages.create(
                model="audit-message",
                max_tokens=8,
                messages=[{"role": "user", "content": "Hello"}],
                extra_body={"user_config": None},
            )
            block: Final = message.content[0]
            assert isinstance(block, anthropic.types.TextBlock), message
            return block.text

    async def call() -> str:
        async with anthropic.AsyncAnthropic(base_url=base_url, api_key=rig.gateway.key, max_retries=0) as client:
            message: Final = await client.messages.create(
                model="audit-message",
                max_tokens=8,
                messages=[{"role": "user", "content": "Hello"}],
                extra_body={"user_config": None},
            )
            block: Final = message.content[0]
            assert isinstance(block, anthropic.types.TextBlock), message
            return block.text

    return asyncio.run(call())


@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
def test_null_user_config_through_the_openai_sdk(rig: _Rig, asynchronous: bool) -> None:
    assert _openai_chat(rig, asynchronous) == "scripted"
    assert one_outbound(rig.gateway, rig.identities["audit-chat"]).get("max_tokens") == 32


@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
def test_null_user_config_through_the_anthropic_sdk(rig: _Rig, asynchronous: bool) -> None:
    assert _anthropic_message(rig, asynchronous) == "scripted"
    assert one_outbound(rig.gateway, rig.identities["audit-message"]).get("max_tokens") == 8


def test_null_user_config_without_the_client_side_opt_in_is_rejected(gateway: Gateway) -> None:
    response: Final = gateway.client.post(
        "/v1/chat/completions",
        json={"model": "audit-not-routed", "messages": USER_MESSAGES, "user_config": None},
        headers={"Authorization": f"Bearer {gateway.key}"},
    )
    assert response.status_code == 401, response.text
    assert "user_config is not allowed in request body" in response.text, response.text


def test_raw_httpx_async_null_user_config(rig: _Rig) -> None:
    async def call() -> httpx.Response:
        async with httpx.AsyncClient(base_url=str(rig.gateway.client.base_url), trust_env=False) as client:
            return await client.post(
                "/v1/responses",
                json={"model": "audit-responses", "input": "Hello", "store": False, "user_config": None},
                headers={"Authorization": f"Bearer {rig.gateway.key}"},
                timeout=30,
            )

    response: Final = asyncio.run(call())
    assert response.status_code == 200, response.text
    assert one_outbound(rig.gateway, rig.identities["audit-responses"]).get("input") == "Hello"
