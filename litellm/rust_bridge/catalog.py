"""Which implementation serves a call.

``POLICIES`` is the rollout: one pure policy per ported route returning ``Required()`` where no
Python implementation exists, ``OptIn()`` where the global switch decides, or ``Python(reason)``
naming the gap that keeps a call on Python. A route without a policy is not ported. ``decide``
applies the switch to that rollout.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, TypeAlias

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
class OptIn:
    pass


@dataclass(frozen=True, slots=True)
class Required:
    pass


@dataclass(frozen=True, slots=True)
class Rust:
    required: bool = False


Rollout: TypeAlias = Python | OptIn | Required
Decision: TypeAlias = Python | Rust
Policy: TypeAlias = Callable[[RouteContext], Rollout]


def opt_in(context: RouteContext) -> Rollout:
    return OptIn()


def required(context: RouteContext) -> Rollout:
    return Required()


def _transcription(context: RouteContext) -> Rollout:
    if context.provider != "bedrock":
        return Python("only Bedrock transcription is ported")
    return Required()


def _messages(context: RouteContext) -> Rollout:
    if context.provider != "anthropic":
        return Python("only Anthropic Messages is ported")
    return OptIn()


POLICIES: Final[Mapping[Route, Policy]] = MappingProxyType(
    {
        Route.OCR: required,
        Route.TRANSCRIPTION: _transcription,
        Route.MESSAGES: _messages,
    }
)


def rollout(context: RouteContext, policies: Mapping[Route, Policy] = POLICIES) -> Rollout:
    policy: Final = policies.get(context.route)
    if policy is None:
        return Python(f"{context.route.value} is not ported")
    return policy(context)


def _resolve(selected: Rollout, *, rust_enabled: bool) -> Decision:
    match selected:
        case Python():
            return selected
        case OptIn():
            return Rust() if rust_enabled else Python("Rust is switched off")
        case Required():
            return Rust(required=True)
        case _:
            assert_never(selected)


def decide(context: RouteContext, policies: Mapping[Route, Policy] = POLICIES) -> Decision:
    return _resolve(rollout(context, policies), rust_enabled=rust_enabled())


def logger() -> Decision:
    return _resolve(OptIn(), rust_enabled=rust_enabled())
