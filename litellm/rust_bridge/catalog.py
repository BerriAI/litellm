"""Which implementation serves a call.

``POLICIES`` is the rollout: one policy per ported route, free to read the context and process
state, returning ``Rust(required=True)`` where no Python implementation exists, ``optional()``
where the global switch decides, or ``Python(reason)`` naming the gap that keeps a call on
Python. ``decide`` only looks the route up; a route without a policy is not ported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, TypeAlias

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


def required(context: RouteContext) -> Decision:
    return Rust(required=True)


def _transcription(context: RouteContext) -> Decision:
    if context.provider != "bedrock":
        return Python("only Bedrock transcription is ported")
    return Rust(required=True)


def _messages(context: RouteContext) -> Decision:
    if context.provider != "anthropic":
        return Python("only Anthropic Messages is ported")
    return optional()


POLICIES: Final[Mapping[Route, Policy]] = MappingProxyType(
    {
        Route.OCR: required,
        Route.TRANSCRIPTION: _transcription,
        Route.MESSAGES: _messages,
    }
)


def decide(context: RouteContext, policies: Mapping[Route, Policy] = POLICIES) -> Decision:
    policy: Final = policies.get(context.route)
    if policy is None:
        return Python(f"{context.route.value} is not ported")
    return policy(context)


def logger() -> Decision:
    return optional()
