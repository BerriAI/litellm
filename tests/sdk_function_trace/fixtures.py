from __future__ import annotations

import base64
import io
import json
import wave
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol, cast

from tests.sdk_function_trace.mock_provider import MockProviderResponse
from tests.sdk_function_trace.steps import Engine

ANTHROPIC_MODEL: Final = "claude-sonnet-5"
OCR_MODEL: Final = "mistral-ocr-latest"
AUDIO_MODEL: Final = "mistral.voxtral-mini-3b-2507"


class SdkCall(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


@dataclass(frozen=True, slots=True)
class Fixture:
    kwargs: dict[str, object]
    provider_response: MockProviderResponse


@dataclass(frozen=True, slots=True)
class RouteSpec:
    label: str
    python_entrypoints: tuple[str, str]
    rust_entrypoints: tuple[str, str]
    fixture: Callable[[Engine], Fixture]


@dataclass(frozen=True, slots=True)
class Invocation:
    function: SdkCall
    kwargs: dict[str, object]
    provider_response: MockProviderResponse
    label: str


def audio_bytes() -> bytes:
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\x00\x00" * 1600)
        return buffer.getvalue()


def _anthropic_message_response() -> MockProviderResponse:
    body: Final = {
        "id": "msg_trace",
        "type": "message",
        "role": "assistant",
        "model": ANTHROPIC_MODEL,
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 2, "output_tokens": 3},
    }
    return MockProviderResponse(200, (("content-type", "application/json"),), json.dumps(body).encode())


def _conversation() -> dict[str, object]:
    return {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16}


def _ocr_fixture(engine: Engine) -> Fixture:
    return Fixture(
        kwargs={
            "model": f"mistral/{OCR_MODEL}",
            "document": {"type": "document_url", "document_url": "https://example.com/document.pdf"},
            **({"optional_params": {"pages": [0]}} if engine == "rust" else {"pages": [0]}),
        },
        provider_response=MockProviderResponse(
            200,
            (("content-type", "application/json"),),
            json.dumps(
                {
                    "pages": [{"index": 0, "markdown": "hello"}],
                    "model": OCR_MODEL,
                    "usage_info": {"pages_processed": 1},
                }
            ).encode(),
        ),
    )


def _chat_completions_fixture(engine: Engine) -> Fixture:
    conversation: Final = _conversation()
    payload: Final = (
        {"messages": conversation["messages"], "optional_params": {"max_tokens": 16}}
        if engine == "rust"
        else conversation
    )
    return Fixture(
        kwargs={"model": f"anthropic/{ANTHROPIC_MODEL}", **payload},
        provider_response=_anthropic_message_response(),
    )


def _messages_fixture(engine: Engine) -> Fixture:
    conversation: Final = _conversation()
    payload: Final = {"body": {**conversation, "model": ANTHROPIC_MODEL}} if engine == "rust" else conversation
    return Fixture(
        kwargs={"model": f"anthropic/{ANTHROPIC_MODEL}", **payload},
        provider_response=_anthropic_message_response(),
    )


def _transcription_fixture(engine: Engine) -> Fixture:
    credentials: Final = {
        "aws_access_key_id": "test-access",
        "aws_secret_access_key": "test-secret",
        "aws_region_name": "us-east-1",
    }
    payload: Final = (
        {
            "audio": {"data": base64.b64encode(audio_bytes()).decode(), "format": "wav"},
            "optional_params": credentials,
        }
        if engine == "rust"
        else {"file": ("sample.wav", audio_bytes(), "audio/wav"), **credentials}
    )
    return Fixture(
        kwargs={"model": f"bedrock/{AUDIO_MODEL}", **payload},
        provider_response=MockProviderResponse(
            200,
            (("content-type", "application/json"),),
            json.dumps(
                {
                    "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5},
                }
            ).encode(),
        ),
    )


ROUTE_SPECS: Final[dict[str, RouteSpec]] = {
    "chat_completions": RouteSpec(
        label="anthropic",
        python_entrypoints=("completion", "acompletion"),
        rust_entrypoints=("chat_completions", "achat_completions"),
        fixture=_chat_completions_fixture,
    ),
    "audio_transcription": RouteSpec(
        label="bedrock (Rust-only provider; Python trace covers SDK dispatch)",
        python_entrypoints=("transcription", "atranscription"),
        rust_entrypoints=("transcription", "atranscription"),
        fixture=_transcription_fixture,
    ),
    "messages": RouteSpec(
        label="anthropic",
        python_entrypoints=("create", "acreate"),
        rust_entrypoints=("messages", "amessages"),
        fixture=_messages_fixture,
    ),
    "ocr": RouteSpec(
        label="mistral",
        python_entrypoints=("ocr", "aocr"),
        rust_entrypoints=("ocr", "aocr"),
        fixture=_ocr_fixture,
    ),
}

ROUTES: Final = tuple(ROUTE_SPECS)


def sdk_invocation(route: str, *, engine: Engine, asynchronous: bool) -> Invocation:
    import litellm
    from litellm.anthropic_interface import messages as sdk_messages
    from litellm.rust_bridge import get_native_bridge

    rust: Final = engine == "rust"
    bridge: Final = get_native_bridge() if rust else None
    if rust and bridge is None:
        raise RuntimeError("Build the native extension first: maturin develop")
    spec: Final = ROUTE_SPECS.get(route)
    if spec is None:
        raise ValueError(f"Unknown route: {route}")
    fixture: Final = spec.fixture(engine)
    owner: Final = bridge if rust else (sdk_messages if route == "messages" else litellm)
    entrypoint: Final = (spec.rust_entrypoints if rust else spec.python_entrypoints)[int(asynchronous)]
    return Invocation(
        function=cast(SdkCall, getattr(owner, entrypoint)),
        kwargs={
            **fixture.kwargs,
            "api_key": "test-key",
            **({"trace": True, "timeout_seconds": 5} if rust else {"timeout": 5}),
        },
        provider_response=fixture.provider_response,
        label=spec.label,
    )
