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
from typing import Final, Protocol, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter

from tests.sdk_function_trace.mock_provider import MockProviderResponse, mock_provider
from tests.sdk_function_trace.profiler import FunctionTraceEvent, profile_python
from tests.sdk_function_trace.steps import Engine, pipeline_issues, pipeline_steps

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


@dataclass(frozen=True, slots=True)
class TraceDiff:
    python_only: tuple[str, ...]
    rust_only: tuple[str, ...]
    shared_order_matches: bool


@dataclass(frozen=True, slots=True)
class TraceRun:
    events: tuple[FunctionTraceEvent, ...] = ()
    error: str | None = None
    skipped: bool = False


def attempt_trace(route: str, *, engine: Engine, asynchronous: bool) -> TraceRun:
    try:
        return TraceRun(events=run_trace(route, rust=engine == "rust", asynchronous=asynchronous))
    except Exception as error:
        return TraceRun(
            error=f"{type(error).__name__}: {error}",
            skipped=(
                route == "messages"
                and engine == "python"
                and not asynchronous
                and isinstance(error, ValueError)
                and str(error) == "anthropic_messages_handler is not implemented for sync calls"
            ),
        )


def trace_diff(python: tuple[FunctionTraceEvent, ...], rust: tuple[FunctionTraceEvent, ...]) -> TraceDiff:
    python_names: Final = {event.function for event in python}
    rust_names: Final = {event.function for event in rust}
    shared_python: Final = tuple(event.function for event in python if event.function in rust_names)
    shared_rust: Final = tuple(event.function for event in rust if event.function in python_names)
    return TraceDiff(
        python_only=tuple(event.function for event in python if event.function not in rust_names),
        rust_only=tuple(event.function for event in rust if event.function not in python_names),
        shared_order_matches=bool(shared_python) and shared_python == shared_rust,
    )


_PYTHON_ONLY_COLOR: Final = "\033[34m"
_RUST_ONLY_COLOR: Final = "\033[33m"
_RESET: Final = "\033[0m"


def _print_tree(
    events: tuple[FunctionTraceEvent, ...], only: frozenset[str], marker: str, color: str, *, colorize: bool
) -> None:
    for event in events:
        line = f"{'  ' * event.depth}{event.function}" + (f"  {marker}" if event.function in only else "")
        sys.stdout.write(f"{color}{line}{_RESET}\n" if colorize and event.function in only else f"{line}\n")


def report(route: str, *, asynchronous: bool, full: bool) -> bool:
    case: Final = invocation(route, rust=False, asynchronous=asynchronous)
    python: Final = attempt_trace(route, engine="python", asynchronous=asynchronous)
    rust: Final = attempt_trace(route, engine="rust", asynchronous=asynchronous)
    python_steps: Final = pipeline_steps(route, "python", python.events)
    rust_steps: Final = pipeline_steps(route, "rust", rust.events)
    diff: Final = trace_diff(python_steps, rust_steps)
    sys.stdout.write(f"route: {route}    provider: {case.label}    mode: {'async' if asynchronous else 'sync'}\n\n")
    colorize: Final = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    cases: Final[tuple[tuple[Engine, TraceRun, tuple[FunctionTraceEvent, ...]], ...]] = (
        ("python", python, python_steps),
        ("rust", rust, rust_steps),
    )
    issues: Final = tuple(
        (engine, () if result.error else pipeline_issues(route, engine, steps)) for engine, result, steps in cases
    )
    for engine, result, shown, only, color in (
        ("python", python, python.events if full else python_steps, diff.python_only, _PYTHON_ONLY_COLOR),
        ("rust", rust, rust.events if full else rust_steps, diff.rust_only, _RUST_ONLY_COLOR),
    ):
        if result.error:
            sys.stdout.write(f"{engine}: {'SKIP' if result.skipped else 'FAIL'} ({result.error})\n\n")
            continue
        sys.stdout.write(f"{engine} ({len(shown)} steps)\n\n")
        _print_tree(
            shown,
            frozenset() if full or python.error or rust.error else frozenset(only),
            f"<- {engine} only",
            color,
            colorize=colorize,
        )
        sys.stdout.write("\n")
    if not python.error and not rust.error:
        order: Final = "the same" if diff.shared_order_matches else "a different"
        sys.stdout.write("diff\n\n")
        sys.stdout.write(f"shared steps appear in {order} order\n")
        sys.stdout.write(f"python-only: {', '.join(diff.python_only) or 'none'}\n")
        sys.stdout.write(f"rust-only: {', '.join(diff.rust_only) or 'none'}\n\n")
    for (checked_engine, checked_result, _), (_, problems) in zip(cases, issues):
        if checked_result.error:
            continue
        sys.stdout.write(
            f"{checked_engine} "
            f"{'SDK dispatch only' if route == 'audio_transcription' and checked_engine == 'python' else 'pipeline'}: "
            f"{'FAIL: ' + '; '.join(problems) if problems else 'PASS'}\n"
        )
    sys.stdout.write("Each successful invocation issued exactly one local provider request\n\n")
    return not any(problems for _, problems in issues) and all(
        result.error is None or result.skipped for result in (python, rust)
    )


def main() -> None:
    parser: Final = argparse.ArgumentParser(description="Compare Python and Rust SDK pipeline steps per route")
    parser.add_argument("--route", choices=("all", *ROUTES), default="all")
    mode: Final = parser.add_mutually_exclusive_group()
    mode.add_argument("--async", dest="asynchronous", action="store_true", default=True)
    mode.add_argument("--sync", dest="asynchronous", action="store_false")
    mode.add_argument("--both", action="store_true", help="run async and sync for every selected route")
    parser.add_argument("--check", action="store_true", help="exit nonzero for missing or misordered pipeline steps")
    parser.add_argument(
        "--full", action="store_true", help="print every captured runtime event instead of pipeline steps"
    )
    args: Final = parser.parse_args()
    route: Final = TypeAdapter(str).validate_python(vars(args)["route"], strict=True)
    asynchronous: Final = TypeAdapter(bool).validate_python(vars(args)["asynchronous"], strict=True)
    full: Final = TypeAdapter(bool).validate_python(vars(args)["full"], strict=True)
    both: Final = TypeAdapter(bool).validate_python(vars(args)["both"], strict=True)
    check: Final = TypeAdapter(bool).validate_python(vars(args)["check"], strict=True)
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    results: Final = tuple(
        report(selected, asynchronous=selected_mode, full=full)
        for selected in ROUTES
        if route in ("all", selected)
        for selected_mode in ((True, False) if both else (asynchronous,))
    )
    if check and not all(results):
        raise SystemExit(1)
