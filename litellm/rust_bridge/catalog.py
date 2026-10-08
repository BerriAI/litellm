"""Which implementation serves a call.

``decide`` is the whole rollout policy: one branch per route, each naming the gap
that keeps a call on Python. A required Rust route has no Python implementation.
An optional one follows the global switch in ``configuration``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from typing_extensions import assert_never

from litellm.rust_bridge.configuration import rust_enabled


class Route(str, Enum):
    CHAT_COMPLETIONS = "chat_completions"
    EMBEDDINGS = "embeddings"
    MESSAGES = "messages"
    RESPONSES = "responses"
    TRANSCRIPTION = "transcription"
    OCR = "ocr"
    TOKEN_COUNTER = "token_counter"
    TOKENIZER = "tokenizer"


@dataclass(frozen=True, slots=True)
class RouteContext:
    route: Route
    provider: str | None = None
    model: str | None = None


@dataclass(frozen=True, slots=True)
class Python:
    reason: str


@dataclass(frozen=True, slots=True)
class Rust:
    required: bool = False


Decision: TypeAlias = Python | Rust
Policy: TypeAlias = Callable[[RouteContext], Decision]


def optional() -> Decision:
    return Rust() if rust_enabled() else Python("Rust is switched off")


def decide(context: RouteContext) -> Decision:
    match context.route:
        case Route.OCR:
            return Rust(required=True)
        case Route.TRANSCRIPTION:
            if context.provider != "bedrock":
                return Python("only Bedrock transcription is ported")
            return Rust(required=True)
        case Route.MESSAGES:
            if context.provider != "anthropic":
                return Python("only Anthropic Messages is ported")
            return optional()
        case Route.CHAT_COMPLETIONS | Route.EMBEDDINGS | Route.RESPONSES | Route.TOKEN_COUNTER | Route.TOKENIZER:
            return Python(f"{context.route.value} is not ported")
        case _:
            assert_never(context.route)


def logger() -> Decision:
    return optional()
