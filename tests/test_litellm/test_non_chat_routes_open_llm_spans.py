"""Regression tests: every route that issues an upstream call must fire the
``pre_call`` input hook.

Tracing integrations open their LLM-call span there (``OpenTelemetryV2`` keys the
span off ``log_pre_api_call`` and treats "no pre_call" as "the request never
reached a provider"), so a handler that skips it leaves the call with no LLM-call
span in the trace at all. Speech, async image generation and moderation each used
to skip it.
"""

import asyncio
from typing import Any, Final

import httpx
import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI

import litellm
from litellm.integrations.custom_logger import CustomLogger


class _PreCallRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.call_types: list[str] = []  # mutable-ok: test recorder of hook calls
        self.api_bases: list[str] = []  # mutable-ok: test recorder of hook calls
        self.request_bodies: list[Any] = []  # mutable-ok: test recorder of hook calls

    def log_pre_api_call(self, model, messages, kwargs) -> None:
        self.call_types.append(str(kwargs.get("call_type")))
        self.api_bases.append(str(kwargs.get("litellm_params", {}).get("api_base")))
        self.request_bodies.append(kwargs.get("additional_args", {}).get("complete_input_dict"))


class _FakeSpeech:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []  # mutable-ok: test recorder of SDK calls

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        request: Final = httpx.Request("POST", "https://api.openai.com/v1/audio/speech")
        return type(
            "_Speech",
            (),
            {"response": httpx.Response(200, content=b"audio-bytes", request=request)},
        )()


class _FakeImages:
    async def generate(self, **kwargs: Any) -> Any:
        return type(
            "_Images",
            (),
            {
                "model_dump": lambda self: {
                    "created": 1,
                    "data": [{"url": "https://example.com/img.png"}],
                }
            },
        )()


class _FakeModerations:
    async def create(self, **kwargs: Any) -> Any:
        return type(
            "_Moderations",
            (),
            {
                "model_dump": lambda self: {
                    "id": "modr-1",
                    "model": "omni-moderation-latest",
                    "results": [
                        {
                            "flagged": False,
                            "categories": {},
                            "category_scores": {},
                            "category_applied_input_types": {},
                        }
                    ],
                }
            },
        )()


class _FakeAsyncOpenAI(AsyncOpenAI):
    """Stands in for the injected client: a real ``AsyncOpenAI`` (``amoderation``
    type-checks it) whose resource namespaces answer without a network call."""

    def __init__(self, base_url: str = "https://api.openai.com/v1") -> None:
        super().__init__(api_key="sk-test", base_url=base_url)
        self.speech = _FakeSpeech()
        self.audio = type("_Audio", (), {"speech": self.speech})()
        self.images = _FakeImages()
        self.moderations = _FakeModerations()


class _FakeAsyncAzureOpenAI(AsyncAzureOpenAI):
    """Same idea for the Azure entrypoint, which resolves no default endpoint of
    its own when ``AZURE_API_BASE`` is unset."""

    def __init__(self) -> None:
        super().__init__(
            api_key="sk-test",
            api_version="2024-02-01",
            azure_endpoint="https://unit-test.openai.azure.com",
        )
        self.speech = _FakeSpeech()
        self.audio = type("_Audio", (), {"speech": self.speech})()


@pytest.fixture
def recorder(monkeypatch):
    recorder: Final = _PreCallRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(litellm, "success_callback", [])
    return recorder


def test_async_speech_opens_an_llm_span(recorder):
    asyncio.run(
        litellm.aspeech(
            model="openai/tts-1",
            input="hello",
            voice="alloy",
            client=_FakeAsyncOpenAI(),
        )
    )
    assert recorder.call_types == ["aspeech"]


def test_azure_async_speech_opens_an_llm_span_without_api_base(recorder, monkeypatch):
    """Azure resolves no default endpoint, so a missing ``api_base`` used to reach
    ``_get_masked_api_base`` as ``None``; the ``TypeError`` was swallowed and the
    whole callback dispatch was skipped."""
    monkeypatch.delenv("AZURE_API_BASE", raising=False)
    asyncio.run(
        litellm.aspeech(
            model="azure/tts-deployment",
            input="hello",
            voice="alloy",
            client=_FakeAsyncAzureOpenAI(),
        )
    )
    assert recorder.call_types == ["aspeech"]
    assert recorder.api_bases == ["https://unit-test.openai.azure.com/openai/"]


def test_azure_async_speech_keeps_caller_headers_out_of_the_logged_body(recorder):
    """The Azure entrypoint carries caller headers in ``optional_params``, so they reach
    the provider as a request kwarg; telemetry reads the logged body, which must stay
    free of them."""
    headers: Final = {"authorization": "Bearer caller-secret"}
    client: Final = _FakeAsyncAzureOpenAI()
    asyncio.run(
        litellm.aspeech(
            model="azure/tts-deployment",
            input="hello",
            voice="alloy",
            extra_headers=headers,
            client=client,
        )
    )
    assert recorder.call_types == ["aspeech"]
    assert "extra_headers" not in recorder.request_bodies[0]
    assert client.speech.calls[0]["extra_headers"] == headers


def test_async_image_generation_opens_an_llm_span(recorder):
    asyncio.run(
        litellm.aimage_generation(
            model="openai/dall-e-3",
            prompt="a cat",
            client=_FakeAsyncOpenAI(),
        )
    )
    assert recorder.call_types == ["aimage_generation"]


def test_async_moderation_opens_an_llm_span(recorder):
    asyncio.run(
        litellm.amoderation(
            model="omni-moderation-latest",
            input="hello",
            client=_FakeAsyncOpenAI(base_url="https://gateway.example/v1"),
        )
    )
    assert recorder.call_types == ["amoderation"]
    assert recorder.api_bases == ["https://gateway.example/v1/"]
