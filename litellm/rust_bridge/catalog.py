"""Declarative Rust/Python selection matrix for every public LiteLLM route.

Rules are static data matched top to bottom; the first match wins and a
context with no matching rule stays on Python. Whether the Rust core can serve
a specific request body is not decided here: that is Rust admission, which
signals ``RustBridgeDeclined`` before any provider I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import Final, TypeAlias

from litellm.rust_bridge.configuration import Decision, Rollout
from litellm.rust_bridge.configuration import decision as _decision


class Route(StrEnum):
    CHAT_COMPLETIONS = "chat_completions"
    MESSAGES = "messages"
    RESPONSES = "responses"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDIT = "image_edit"
    SPEECH = "speech"
    TRANSCRIPTION = "transcription"
    MODERATION = "moderation"
    OCR = "ocr"


class Delivery(Enum):
    COMPLETED = auto()
    STREAMING = auto()
    WEBSOCKET = auto()


@dataclass(frozen=True, slots=True)
class Context:
    route: Route
    provider: str | None = None
    model: str | None = None
    delivery: Delivery = Delivery.COMPLETED


@dataclass(frozen=True, slots=True)
class Rule:
    route: Route
    rollout: Rollout
    providers: frozenset[str] | None = None
    models: frozenset[str] | None = None
    deliveries: frozenset[Delivery] | None = None

    def matches(self, context: Context) -> bool:
        return (
            context.route is self.route
            and (self.providers is None or context.provider in self.providers)
            and (self.models is None or context.model in self.models)
            and (self.deliveries is None or context.delivery in self.deliveries)
        )


Rules: TypeAlias = tuple[Rule, ...]

_COMPLETED: Final = frozenset({Delivery.COMPLETED})

RULES: Final[Rules] = (
    Rule(Route.OCR, Rollout.RUST_OPT_OUT),
    Rule(Route.TRANSCRIPTION, Rollout.RUST_REQUIRED, providers=frozenset({"bedrock"})),
    Rule(Route.TRANSCRIPTION, Rollout.PYTHON_ONLY),
    Rule(
        Route.CHAT_COMPLETIONS,
        Rollout.RUST_OPT_IN,
        providers=frozenset({"anthropic", "bedrock"}),
        deliveries=_COMPLETED,
    ),
    Rule(Route.CHAT_COMPLETIONS, Rollout.PYTHON_ONLY),
    Rule(Route.MESSAGES, Rollout.RUST_OPT_IN, providers=frozenset({"anthropic", "azure_ai"})),
    Rule(Route.MESSAGES, Rollout.PYTHON_ONLY),
    Rule(
        Route.RESPONSES,
        Rollout.RUST_OPT_IN,
        providers=frozenset({"openai"}),
        deliveries=frozenset({Delivery.WEBSOCKET}),
    ),
    Rule(Route.RESPONSES, Rollout.PYTHON_ONLY),
    Rule(Route.EMBEDDING, Rollout.PYTHON_ONLY),
    Rule(Route.RERANK, Rollout.PYTHON_ONLY),
    Rule(Route.IMAGE_GENERATION, Rollout.PYTHON_ONLY),
    Rule(Route.IMAGE_EDIT, Rollout.PYTHON_ONLY),
    Rule(Route.SPEECH, Rollout.PYTHON_ONLY),
    Rule(Route.MODERATION, Rollout.PYTHON_ONLY),
)


def rollout(context: Context, rules: Rules = RULES) -> Rollout:
    return next((rule.rollout for rule in rules if rule.matches(context)), Rollout.PYTHON_ONLY)


def decision(context: Context, rules: Rules = RULES) -> Decision:
    return _decision(rollout(context, rules))
