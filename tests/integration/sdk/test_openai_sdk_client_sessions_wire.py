import asyncio
import base64
import json
import struct
import threading
import zlib
from collections.abc import Awaitable, Callable, Iterator, Mapping
from queue import SimpleQueue
from typing import Final, TypeVar
from urllib.parse import urlsplit

import httpx
import openai
import pytest
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

import litellm
from litellm import Router
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.utils import ImageResponse, ModelResponse, TextCompletionResponse

_R: Final = TypeVar("_R")

_PROVIDER_KEY: Final = "sk-scripted-provider"
_API_VERSION: Final = "2024-10-21"
_GATEWAY_PATH: Final = "/gateway.ai.cloudflare.com/v1/scripted-account/scripted-gateway/azure-openai/scripted-resource"
_DEPLOYMENT: Final = "gpt-4o-mini-gateway"
_ANSWER: Final = "wire answer"
_IMAGE_URL: Final = "https://images.example.invalid/variation.png"


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


_PNG: Final = (
    b"\x89PNG\r\n\x1a\n"
    + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    + _png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
    + _png_chunk(b"IEND", b"")
)


def _json(body: Mapping[str, JsonValue]) -> Reply:
    return Reply(body=json.dumps(body).encode())


def _chat_completion() -> Mapping[str, JsonValue]:
    return {
        "id": "chatcmpl-wire",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": _ANSWER}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _text_completion() -> Mapping[str, JsonValue]:
    return {
        "id": "cmpl-wire",
        "object": "text_completion",
        "created": 1,
        "model": "gpt-3.5-turbo-instruct",
        "choices": [{"index": 0, "text": _ANSWER, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _peer(expected_path: str, body: Mapping[str, JsonValue]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert urlsplit(request.target).path == expected_path, request.target
        return _json(body)

    return respond


def _held_peer(gate: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert gate.wait(timeout=10), "the timeout cell never released its peer"
        return _json(_chat_completion())

    return respond


def _only_request(wire: Wire) -> Request:
    received: Final = wire.drain()
    assert len(received) == 1, received
    return received[0]


def _drain(seen: SimpleQueue[str]) -> tuple[str, ...]:
    return tuple(seen.get_nowait() for _ in range(seen.qsize()))


def _content(response: ModelResponse) -> str | None:
    choice: Final = response.choices[0]
    return choice.message.content if isinstance(choice, litellm.Choices) else None


@pytest.fixture
def sync_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[httpx.Client, SimpleQueue[str]]]:
    seen: Final[SimpleQueue[str]] = SimpleQueue()

    def record(request: httpx.Request) -> None:
        seen.put(str(request.url))

    with httpx.Client(event_hooks={"request": [record]}) as client:
        monkeypatch.setattr(litellm, "client_session", client)
        monkeypatch.setattr(litellm, "aclient_session", None)
        yield client, seen


def _run_with_async_session(
    monkeypatch: pytest.MonkeyPatch, call: Callable[[], Awaitable[_R]]
) -> tuple[_R, tuple[str, ...]]:
    seen: Final[SimpleQueue[str]] = SimpleQueue()

    async def record(request: httpx.Request) -> None:
        seen.put(str(request.url))

    async def run() -> _R:
        async with httpx.AsyncClient(event_hooks={"request": [record]}) as client:
            monkeypatch.setattr(litellm, "client_session", None)
            monkeypatch.setattr(litellm, "aclient_session", client)
            return await call()

    result: Final = asyncio.run(run())
    return result, _drain(seen)


def _deployment(timeout: httpx.Timeout | openai.Timeout, api_base: str, deployment_id: str) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": api_base,
                    "api_key": _PROVIDER_KEY,
                    "timeout": timeout,
                    "max_retries": 0,
                },
                "model_info": {"id": deployment_id},
            }
        ],
        num_retries=0,
    )


def test_f01_async_image_variation_with_only_a_sync_session_set_builds_its_own_async_client(
    sync_session: tuple[httpx.Client, SimpleQueue[str]],
) -> None:
    with wire_server(_peer("/images/variations", {"created": 1, "data": [{"url": _IMAGE_URL}]})) as wire:
        response: Final = asyncio.run(
            litellm.aimage_variation(
                image=("probe.png", _PNG, "image/png"),
                model="dall-e-2",
                custom_llm_provider="openai",
                api_base=wire.url,
                api_key=_PROVIDER_KEY,
                num_retries=0,
            )
        )
        assert isinstance(response, ImageResponse), type(response)
        assert response.data is not None and response.data[0].url == _IMAGE_URL, response
        request: Final = _only_request(wire)
        assert request.method == "POST", request.method
        assert _PNG in request.body
        assert request.headers.get("authorization") == f"Bearer {_PROVIDER_KEY}", request.headers
        assert _drain(sync_session[1]) == ()


def test_f02_async_azure_cloudflare_gateway_call_with_only_a_sync_session_set_builds_its_own_async_client(
    sync_session: tuple[httpx.Client, SimpleQueue[str]],
) -> None:
    with wire_server(_peer(f"{_GATEWAY_PATH}/{_DEPLOYMENT}/chat/completions", _chat_completion())) as wire:
        response: Final = asyncio.run(
            litellm.acompletion(
                model=f"azure/{_DEPLOYMENT}",
                messages=[{"role": "user", "content": "via the gateway"}],
                api_base=f"{wire.url}{_GATEWAY_PATH}",
                api_key=_PROVIDER_KEY,
                api_version=_API_VERSION,
                num_retries=0,
            )
        )
        assert isinstance(response, ModelResponse), type(response)
        assert _content(response) == _ANSWER, response
        request: Final = _only_request(wire)
        assert urlsplit(request.target).query == f"api-version={_API_VERSION}", request.target
        assert request.headers.get("api-key") == _PROVIDER_KEY, request.headers
        assert _drain(sync_session[1]) == ()


def test_f03_async_text_completion_uses_the_callers_async_session(monkeypatch: pytest.MonkeyPatch) -> None:
    with wire_server(_peer("/completions", _text_completion())) as wire:
        response, seen = _run_with_async_session(
            monkeypatch,
            lambda: litellm.atext_completion(
                model="gpt-3.5-turbo-instruct",
                prompt="complete this",
                custom_llm_provider="openai",
                api_base=wire.url,
                api_key=_PROVIDER_KEY,
                num_retries=0,
            ),
        )
        assert isinstance(response, TextCompletionResponse), type(response)
        assert response.choices[0].text == _ANSWER, response
        assert _only_request(wire).method == "POST"
        assert seen == (f"{wire.url}/completions",)


def test_f04_sync_text_completion_uses_the_callers_sync_session(
    sync_session: tuple[httpx.Client, SimpleQueue[str]],
) -> None:
    with wire_server(_peer("/completions", _text_completion())) as wire:
        response: Final = litellm.text_completion(
            model="gpt-3.5-turbo-instruct",
            prompt="complete this",
            custom_llm_provider="openai",
            api_base=wire.url,
            api_key=_PROVIDER_KEY,
            num_retries=0,
        )
        assert isinstance(response, TextCompletionResponse), type(response)
        assert response.choices[0].text == _ANSWER, response
        assert _only_request(wire).method == "POST"
        assert _drain(sync_session[1]) == (f"{wire.url}/completions",)


def test_f05_sync_moderation_reaches_the_peer_and_returns_its_verdict() -> None:
    verdict: Final[Mapping[str, JsonValue]] = {
        "id": "modr-wire",
        "model": "omni-moderation-latest",
        "results": [{"flagged": True, "categories": {"harassment": True}, "category_scores": {"harassment": 0.91}}],
    }
    with wire_server(_peer("/moderations", verdict)) as wire:
        response: Final = litellm.moderation(
            input="moderate this",
            model="omni-moderation-latest",
            api_key=_PROVIDER_KEY,
            api_base=wire.url,
        )
        assert response.results[0].flagged is True, response
        request: Final = _only_request(wire)
        assert json.loads(request.body) == {"input": "moderate this", "model": "omni-moderation-latest"}, request.body


@pytest.mark.parametrize("timeout_type", (httpx.Timeout, openai.Timeout), ids=("httpx", "openai"))
def test_f06_sync_completion_accepts_a_structured_timeout(
    timeout_type: type[httpx.Timeout] | type[openai.Timeout],
) -> None:
    with wire_server(_peer("/chat/completions", _chat_completion())) as wire:
        response: Final = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "with a structured timeout"}],
            custom_llm_provider="openai",
            api_base=wire.url,
            api_key=_PROVIDER_KEY,
            timeout=timeout_type(5.0, connect=2.0),
            num_retries=0,
        )
        assert isinstance(response, ModelResponse), type(response)
        assert _content(response) == _ANSWER, response
        assert _only_request(wire).method == "POST"


def test_f07_async_azure_cloudflare_gateway_call_uses_the_callers_async_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with wire_server(_peer(f"{_GATEWAY_PATH}/{_DEPLOYMENT}/chat/completions", _chat_completion())) as wire:
        response, seen = _run_with_async_session(
            monkeypatch,
            lambda: litellm.acompletion(
                model=f"azure/{_DEPLOYMENT}",
                messages=[{"role": "user", "content": "via the gateway"}],
                api_base=f"{wire.url}{_GATEWAY_PATH}",
                api_key=_PROVIDER_KEY,
                api_version=_API_VERSION,
                num_retries=0,
            ),
        )
        assert isinstance(response, ModelResponse), type(response)
        assert _content(response) == _ANSWER, response
        assert _only_request(wire).headers.get("api-key") == _PROVIDER_KEY
        assert seen == (f"{wire.url}{_GATEWAY_PATH}/{_DEPLOYMENT}/chat/completions?api-version={_API_VERSION}",)


def test_f08_sync_image_variation_uses_the_callers_sync_session(
    sync_session: tuple[httpx.Client, SimpleQueue[str]],
) -> None:
    with wire_server(_peer("/images/variations", {"created": 1, "data": [{"url": _IMAGE_URL}]})) as wire:
        response: Final = litellm.image_variation(
            image=("probe.png", _PNG, "image/png"),
            model="dall-e-2",
            custom_llm_provider="openai",
            api_base=wire.url,
            api_key=_PROVIDER_KEY,
            num_retries=0,
        )
        assert isinstance(response, ImageResponse), type(response)
        assert response.data is not None and response.data[0].url == _IMAGE_URL, response
        assert _PNG in _only_request(wire).body
        assert _drain(sync_session[1]) == (f"{wire.url}/images/variations",)


@pytest.mark.parametrize("timeout_type", (httpx.Timeout, openai.Timeout), ids=("httpx", "openai"))
def test_f09_router_deployment_keeps_a_structured_timeout_as_httpx_and_serves(
    timeout_type: type[httpx.Timeout] | type[openai.Timeout],
) -> None:
    with wire_server(_peer("/chat/completions", _chat_completion())) as wire:
        router: Final = _deployment(timeout_type(5.0, connect=2.0), wire.url, "deployment-f09")
        response: Final = router.completion(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "through the router"}]
        )
        assert isinstance(response, ModelResponse), type(response)
        assert _content(response) == _ANSWER, response
        assert _only_request(wire).method == "POST"
        deployment: Final = router.get_deployment("deployment-f09")
        assert deployment is not None
        stored: Final = deployment.litellm_params.timeout
        assert isinstance(stored, httpx.Timeout), type(stored)
        assert (stored.read, stored.connect) == (5.0, 2.0), stored


def test_f10_router_deployment_structured_read_timeout_fires_once_as_a_408() -> None:
    gate: Final = threading.Event()
    with wire_server(_held_peer(gate)) as wire:
        router: Final = _deployment(httpx.Timeout(0.5, connect=2.0), wire.url, "deployment-f10")
        with pytest.raises(litellm.Timeout) as raised:
            router.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "held upstream"}])
        gate.set()
        assert raised.value.status_code == 408, raised.value
        assert len(wire.drain()) == 1


_SPEECH_AUDIO: Final = b"OggS" + bytes(range(60))
_SPEECH_AUDIO_REPLY: Final = Reply(body=_SPEECH_AUDIO, content_type="audio/ogg")
_ELEVENLABS_VOICE: Final = "21m00Tcm4TlvDq8ikWAM"
_VERTEX_PROJECT: Final = "scripted-project"
_SPEECH_ROUTES: Final[Mapping[str, tuple[Mapping[str, str], str, Reply]]] = {
    "openai": (
        {"model": "openai/gpt-4o-mini-tts", "voice": "alloy", "api_key": _PROVIDER_KEY},
        "/audio/speech",
        _SPEECH_AUDIO_REPLY,
    ),
    "azure": (
        {"model": "azure/tts-deployment", "voice": "alloy", "api_key": _PROVIDER_KEY, "api_version": _API_VERSION},
        "/openai/deployments/tts-deployment/audio/speech",
        _SPEECH_AUDIO_REPLY,
    ),
    "azure_ava": (
        {"model": "azure/speech/azure-tts", "voice": "alloy", "api_key": _PROVIDER_KEY},
        "/cognitiveservices/v1",
        _SPEECH_AUDIO_REPLY,
    ),
    "elevenlabs": (
        {"model": "elevenlabs/eleven_multilingual_v2", "voice": _ELEVENLABS_VOICE, "api_key": _PROVIDER_KEY},
        f"/v1/text-to-speech/{_ELEVENLABS_VOICE}",
        _SPEECH_AUDIO_REPLY,
    ),
    "edenai": (
        {"model": "edenai/openai", "voice": "alloy", "api_key": _PROVIDER_KEY},
        "/audio/speech",
        _SPEECH_AUDIO_REPLY,
    ),
    "minimax": (
        {"model": "minimax/speech-02-hd", "voice": "alloy", "api_key": _PROVIDER_KEY},
        "/v1/t2a_v2",
        _json({"data": {"audio": _SPEECH_AUDIO.hex()}, "base_resp": {"status_code": 0, "status_msg": "success"}}),
    ),
    "mistral": (
        {"model": "mistral/voxtral-mini-tts-2603", "voice": "alloy", "api_key": _PROVIDER_KEY},
        "/v1/audio/speech",
        _json({"audio_data": base64.b64encode(_SPEECH_AUDIO).decode()}),
    ),
    "aws_polly": (
        {
            "model": "aws_polly/neural",
            "voice": "Joanna",
            "aws_access_key_id": "AKIASCRIPTED",
            "aws_secret_access_key": "scripted-secret",
            "aws_region_name": "us-east-1",
        },
        "/v1/speech",
        _SPEECH_AUDIO_REPLY,
    ),
}
_PAYMENT_REQUIRED_MODELS: Final[Mapping[str, type[litellm.BadRequestError]]] = {
    "openai/gpt-4o-mini": litellm.BadRequestError,
    "anthropic/claude-sonnet-4-5": litellm.PaymentRequiredError,
}


def _speech_peer(expected_path: str, reply: Reply) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert urlsplit(request.target).path == expected_path, request.target
        return reply

    return respond


def _vertex_speech_peer(request: Request) -> Reply:
    if urlsplit(request.target).path == "/_oauth/token":
        return _json({"access_token": "scripted-vertex-token", "expires_in": 3600, "token_type": "Bearer"})
    return _json({"audioContent": base64.b64encode(_SPEECH_AUDIO).decode()})


def _payment_required_peer(request: Request) -> Reply:
    return Reply(
        status=402,
        body=json.dumps({"error": {"message": "scripted 402", "type": "invalid_request_error"}}).encode(),
    )


@pytest.mark.parametrize("provider", tuple(_SPEECH_ROUTES))
def test_f11_sync_speech_accepts_an_sdk_timeout_on_every_provider_branch(provider: str) -> None:
    route, expected_path, reply = _SPEECH_ROUTES[provider]
    with wire_server(_speech_peer(expected_path, reply)) as wire:
        response: Final = litellm.speech(
            input="speak this",
            api_base=wire.url,
            timeout=openai.Timeout(5.0, connect=2.0),
            **route,
        )
        assert response.content == _SPEECH_AUDIO, response.content[:16]
        assert _only_request(wire).method == "POST"


def test_f13_sync_vertex_speech_accepts_an_sdk_timeout() -> None:
    with wire_server(_vertex_speech_peer) as wire:
        response: Final = litellm.speech(
            model="vertex_ai/chirp",
            voice="alloy",
            input="speak this",
            api_base=wire.url,
            vertex_credentials=service_account_json(_VERTEX_PROJECT, wire.url),
            vertex_project=_VERTEX_PROJECT,
            vertex_location="us-central1",
            timeout=openai.Timeout(5.0, connect=2.0),
        )
        assert response.content == _SPEECH_AUDIO, response.content[:16]
        assert tuple((seen.method, urlsplit(seen.target).path) for seen in wire.drain()) == (
            ("POST", "/_oauth/token"),
            ("POST", "/"),
        )


def _runwayml_rejecting_peer(request: Request) -> Reply:
    assert urlsplit(request.target).path == "/v1/text_to_speech", request.target
    return Reply(status=400, body=json.dumps({"error": "scripted 400"}).encode())


def test_f14_sync_runwayml_speech_sends_its_task_with_an_sdk_timeout() -> None:
    with wire_server(_runwayml_rejecting_peer) as wire:
        with pytest.raises(BaseLLMException) as raised:
            litellm.speech(
                model="runwayml/eleven_multilingual_v2",
                voice="Maya",
                input="speak this",
                api_base=wire.url,
                api_key=_PROVIDER_KEY,
                timeout=openai.Timeout(5.0, connect=2.0),
            )
        assert raised.value.status_code == 400, raised.value
        assert "scripted 400" in raised.value.message, raised.value.message
        assert _only_request(wire).method == "POST"


@pytest.mark.parametrize("model", tuple(_PAYMENT_REQUIRED_MODELS))
def test_f12_sync_completion_maps_an_upstream_402_with_its_message(model: str) -> None:
    with wire_server(_payment_required_peer) as wire:
        with pytest.raises(_PAYMENT_REQUIRED_MODELS[model]) as raised:
            litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "out of credit"}],
                api_base=wire.url,
                api_key=_PROVIDER_KEY,
                num_retries=0,
                max_retries=0,
            )
        assert raised.value.status_code == 402, raised.value
        assert "scripted 402" in raised.value.message, raised.value.message
        assert len(wire.drain()) == 1
