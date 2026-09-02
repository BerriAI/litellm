from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from tests.sdk_function_trace.profiler import FunctionTraceEvent

Engine = Literal["python", "rust"]


@dataclass(frozen=True, slots=True)
class Step:
    name: str
    python: re.Pattern[str] | None
    rust: str | None


def _step(name: str, python: str | None = None, rust: str | None = None) -> Step:
    return Step(name, re.compile(python) if python is not None else None, rust)


_POST: Final = r"AsyncHTTPHandler\.post$|HTTPHandler\.post$"

STEPS: Final[dict[str, tuple[Step, ...]]] = {
    "ocr": (
        _step("ocr", r"ocr/main\.py:\d+ a?ocr$", "ocr"),
        _step("prepare_ocr_call", r"ocr/main\.py:\d+ _prepare_ocr_request$", "prepare_ocr_call"),
        _step("get_provider_ocr_config", r"ProviderConfigManager\.get_provider_ocr_config$"),
        _step("supported_ocr_params", r"get_supported_ocr_params$", "supported_ocr_params"),
        _step("map_ocr_params", r"(?<!async_)map_ocr_params$", "map_ocr_params"),
        _step("validate_environment", r"(?<!_)validate_environment$"),
        _step("complete_url", r"get_complete_url$"),
        _step("transform_ocr_request", r"(?<!async_)transform_ocr_request$", "transform_ocr_request"),
        _step("execute_ocr_provider_call", r"BaseLLMHTTPHandler\.(?:async_)?ocr$", "execute_ocr_provider_call"),
        _step("http_request", _POST),
        _step("transform_ocr_response", r"(?<!async_)transform_ocr_response$", "transform_ocr_response"),
    ),
    "chat_completions": (
        _step("chat_completions", r"main\.py:\d+ a?completion$", "chat_completions"),
        _step("prepare_chat_completions_call", rust="prepare_chat_completions_call"),
        _step("get_provider_chat_config", r"ProviderConfigManager\.get_provider_chat_config$"),
        _step("supported_openai_params", r"get_supported_openai_params$"),
        _step("validate_environment", r"(?<!_)validate_environment$"),
        _step("transform_request", r"(?<!async_)transform_request$", "transform_request"),
        _step(
            "execute_chat_completions_provider_call",
            r"a?completion_function$",
            "execute_chat_completions_provider_call",
        ),
        _step("http_request", _POST),
        _step("transform_response", r"(?<!async_)transform_response$", "transform_response"),
    ),
    "messages": (
        _step("messages", r"anthropic_interface/messages/__init__\.py:\d+ a?create$", "messages"),
        _step("prepare_messages_call", rust="prepare_messages_call"),
        _step("get_provider_messages_config", r"ProviderConfigManager\.get_provider_anthropic_messages_config$"),
        _step("validate_environment", r"validate_anthropic_messages_environment$"),
        _step("transform_request", r"(?<!async_)transform_anthropic_messages_request$", "transform_request"),
        _step("complete_url", r"get_complete_url$"),
        _step(
            "execute_messages_provider_call",
            r"(?:async_)?anthropic_messages_handler$",
            "execute_messages_provider_call",
        ),
        _step("http_request", _POST),
        _step("transform_response", r"(?<!async_)transform_anthropic_messages_response$", "transform_response"),
    ),
    "audio_transcription": (
        _step("audio_transcription", r"main\.py:\d+ a?transcription$", "audio_transcription"),
        _step("prepare_audio_transcription_provider_call", rust="prepare_audio_transcription_provider_call"),
        _step("get_non_default_transcription_params", r"get_non_default_transcription_params$"),
        _step("map_transcription_params", r"get_optional_params_transcription$", "map_transcription_params"),
        _step("get_provider_transcription_config", r"ProviderConfigManager\.get_provider_audio_transcription_config$"),
        _step("supported_transcription_params", rust="supported_transcription_params"),
        _step("transform_transcription_request", rust="transform_transcription_request"),
        _step(
            "execute_audio_transcription_provider_call",
            r"BedrockAudioTranscriptionRustDispatch\.(?:async_)?audio_transcriptions$",
            "execute_audio_transcription_provider_call",
        ),
        _step("transform_transcription_response", rust="transform_transcription_response"),
    ),
}


def _canonical_name(route: str, engine: Engine, function: str) -> str | None:
    for step in STEPS[route]:
        if engine == "python":
            if step.python is not None and step.python.search(function):
                return step.name
        elif step.rust is not None and function == step.rust:
            return step.name
    return function if engine == "rust" else None


def pipeline_steps(route: str, engine: Engine, events: Sequence[FunctionTraceEvent]) -> tuple[FunctionTraceEvent, ...]:
    shown: list[FunctionTraceEvent] = []
    stack: list[tuple[int, int]] = []
    seen: set[str] = set()
    for event in events:
        name = _canonical_name(route, engine, event.function)
        if name is None or name in seen:
            continue
        seen.add(name)
        while stack and event.depth <= stack[-1][0]:
            stack.pop()
        depth = stack[-1][1] + 1 if stack else 0
        stack.append((event.depth, depth))
        shown.append(FunctionTraceEvent(function=name, depth=depth))
    return tuple(shown)
