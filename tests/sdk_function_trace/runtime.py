from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import wave
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter

from tests.sdk_function_trace.mock_provider import MockProviderResponse, mock_provider
from tests.sdk_function_trace.profiler import FunctionTraceEvent, profile_python

ROUTES: Final = ("chat_completions", "audio_transcription", "messages", "ocr")
ANTHROPIC_MODEL: Final = "claude-sonnet-5"
OCR_MODEL: Final = "mistral-ocr-latest"
AUDIO_MODEL: Final = "mistral.voxtral-mini-3b-2507"


class SdkCall(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


class TraceEventPayload(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    function: str
    depth: int


class TraceResponsePayload(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    response: object
    trace: tuple[TraceEventPayload, ...] | list[TraceEventPayload]


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


def request(route: str, *, rust: bool) -> tuple[dict[str, object], dict[str, object]]:
    match route:
        case "ocr":
            return (
                {
                    "model": f"mistral/{OCR_MODEL}",
                    "document": {"type": "document_url", "document_url": "https://example.com/document.pdf"},
                    **({"optional_params": {"pages": [0]}} if rust else {"pages": [0]}),
                },
                {
                    "pages": [{"index": 0, "markdown": "hello"}],
                    "model": OCR_MODEL,
                    "usage_info": {"pages_processed": 1},
                },
            )
        case "chat_completions" | "messages":
            conversation: Final = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16}
            chat_params: Final = (
                {"messages": conversation["messages"], "optional_params": {"max_tokens": 16}} if rust else conversation
            )
            message_params: Final = {"body": {**conversation, "model": ANTHROPIC_MODEL}} if rust else conversation
            return (
                {
                    "model": f"anthropic/{ANTHROPIC_MODEL}",
                    **(chat_params if route == "chat_completions" else message_params),
                },
                {
                    "id": "msg_trace",
                    "type": "message",
                    "role": "assistant",
                    "model": ANTHROPIC_MODEL,
                    "content": [{"type": "text", "text": "hello"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 2, "output_tokens": 3},
                },
            )
        case "audio_transcription":
            credentials: Final = {
                "aws_access_key_id": "test-access",
                "aws_secret_access_key": "test-secret",
                "aws_region_name": "us-east-1",
            }
            return (
                {
                    "model": f"bedrock/{AUDIO_MODEL}",
                    **(
                        {
                            "audio": {"data": base64.b64encode(audio_bytes()).decode(), "format": "wav"},
                            "optional_params": credentials,
                        }
                        if rust
                        else {"file": ("sample.wav", audio_bytes(), "audio/wav"), **credentials}
                    ),
                },
                {
                    "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5},
                },
            )
        case _:
            raise ValueError(f"Unknown route: {route}")


def invocation(route: str, *, rust: bool, asynchronous: bool) -> Invocation:
    import litellm
    from litellm.anthropic_interface import messages as sdk_messages
    from litellm.rust_bridge import get_native_bridge

    bridge: Final = get_native_bridge() if rust else None
    if rust and bridge is None:
        raise RuntimeError("Build the native extension first: maturin develop")
    python_names: Final = {
        "ocr": ("ocr", "aocr"),
        "chat_completions": ("completion", "acompletion"),
        "messages": ("create", "acreate"),
        "audio_transcription": ("transcription", "atranscription"),
    }
    rust_names: Final = {
        "ocr": ("ocr", "aocr"),
        "chat_completions": ("chat_completions", "achat_completions"),
        "messages": ("messages", "amessages"),
        "audio_transcription": ("transcription", "atranscription"),
    }
    labels: Final = {
        "ocr": "mistral",
        "chat_completions": "anthropic",
        "messages": "anthropic",
        "audio_transcription": "bedrock (Rust-only provider; Python trace covers SDK dispatch)",
    }
    kwargs, body = request(route, rust=rust)
    owner: Final = bridge if rust else (sdk_messages if route == "messages" else litellm)
    name: Final = (rust_names if rust else python_names)[route][int(asynchronous)]
    return Invocation(
        function=cast(SdkCall, getattr(owner, name)),
        kwargs={**kwargs, "api_key": "test-key", **({"trace": True, "timeout_seconds": 5} if rust else {"timeout": 5})},
        provider_response=MockProviderResponse(200, (("content-type", "application/json"),), json.dumps(body).encode()),
        label=labels[route],
    )


def collect(case: Invocation, api_base: str, *, rust: bool, asynchronous: bool) -> tuple[FunctionTraceEvent, ...]:
    import litellm

    async def invoke_async() -> object:
        return await cast(Awaitable[object], case.function(**case.kwargs, api_base=api_base))

    if rust:
        raw: Final = asyncio.run(invoke_async()) if asynchronous else case.function(**case.kwargs, api_base=api_base)
        payload: Final = TraceResponsePayload.model_validate(raw)
        return tuple(FunctionTraceEvent(event.function, event.depth) for event in payload.trace)
    with profile_python(source_root=Path(litellm.__file__).parent, threads=True) as profiler:
        if asynchronous:
            asyncio.run(invoke_async())
        else:
            case.function(**case.kwargs, api_base=api_base)
    return tuple(profiler.events)


def run_trace(route: str, *, rust: bool, asynchronous: bool = False) -> tuple[FunctionTraceEvent, ...]:
    from litellm.rust_bridge import ocr as ocr_bridge

    case: Final = invocation(route, rust=rust, asynchronous=asynchronous)
    previous_ocr: Final = ocr_bridge.rust_ocr_enabled()
    previous_rust: Final = os.environ.get("LITELLM_RUST")
    os.environ["LITELLM_RUST"] = "false"
    ocr_bridge.use_litellm_rust(False)
    try:
        with mock_provider(case.provider_response) as api_base:
            events: Final = collect(case, api_base, rust=rust, asynchronous=asynchronous)
        if not events:
            raise RuntimeError(f"No runtime events for {route}; rebuild the native extension with tracing support")
        return events
    finally:
        ocr_bridge.use_litellm_rust(previous_ocr)
        if previous_rust is None:
            os.environ.pop("LITELLM_RUST", None)
        else:
            os.environ["LITELLM_RUST"] = previous_rust


def print_trace(route: str, engine: Literal["python", "rust"], asynchronous: bool) -> None:
    case: Final = invocation(route, rust=engine == "rust", asynchronous=asynchronous)
    events: Final = run_trace(route, rust=engine == "rust", asynchronous=asynchronous)
    sys.stdout.write(f"{engine} {route} | {case.label} | {'async' if asynchronous else 'sync'}\n")
    for event in events:
        sys.stdout.write(f"{'  ' * event.depth}{event.function}\n")
    sys.stdout.write(f"{len(events)} runtime events; one local provider HTTP request\n\n")


def main(engine: Literal["python", "rust"]) -> None:
    parser: Final = argparse.ArgumentParser(description=f"Execute SDK calls and print {engine} runtime function events")
    parser.add_argument("--route", choices=("all", *ROUTES), default="all")
    mode: Final = parser.add_mutually_exclusive_group()
    mode.add_argument("--async", dest="asynchronous", action="store_true", default=True)
    mode.add_argument("--sync", dest="asynchronous", action="store_false")
    args: Final = parser.parse_args()
    route: Final = TypeAdapter(str).validate_python(vars(args)["route"], strict=True)
    asynchronous: Final = TypeAdapter(bool).validate_python(vars(args)["asynchronous"], strict=True)
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for selected in ROUTES:
        if route in ("all", selected):
            print_trace(selected, engine, asynchronous)
