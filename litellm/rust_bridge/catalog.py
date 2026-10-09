"""Ordered rollout policy for routes and loggers.

The first matching rule wins; unmatched contexts stay on Python. Native
admission separately decides whether the selected implementation can execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, TypeAlias

from litellm.rust_bridge.configuration import Decision, Rollout
from litellm.rust_bridge.configuration import decision as _decision


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
class RouteRule:
    route: Route
    rollout: Rollout
    providers: frozenset[str] | None = None
    models: frozenset[str] | None = None

    def matches(self, context: Context) -> bool:
        return (
            isinstance(context, RouteContext)
            and context.route is self.route
            and (self.providers is None or context.provider in self.providers)
            and (self.models is None or context.model in self.models)
        )


@dataclass(frozen=True, slots=True)
class LoggerContext:
    pass


@dataclass(frozen=True, slots=True)
class LoggerRule:
    rollout: Rollout

    def matches(self, context: Context) -> bool:
        return isinstance(context, LoggerContext)


Context: TypeAlias = RouteContext | LoggerContext
Rule: TypeAlias = RouteRule | LoggerRule
Rules: TypeAlias = tuple[Rule, ...]

RULES: Final[Rules] = (
    LoggerRule(Rollout.RUST_OPT_IN),
    RouteRule(Route.CHAT_COMPLETIONS, Rollout.PYTHON_ONLY),
    RouteRule(Route.EMBEDDINGS, Rollout.PYTHON_ONLY),
    RouteRule(Route.OCR, Rollout.RUST_REQUIRED),
    RouteRule(Route.MESSAGES, Rollout.RUST_OPT_IN, providers=frozenset({"anthropic", "vertex_ai"})),
    RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY),
    RouteRule(Route.RESPONSES, Rollout.PYTHON_ONLY),
    RouteRule(Route.TOKEN_COUNTER, Rollout.PYTHON_ONLY),
    RouteRule(Route.TOKENIZER, Rollout.PYTHON_ONLY),
    RouteRule(Route.TRANSCRIPTION, Rollout.RUST_REQUIRED, providers=frozenset({"bedrock"})),
)


def rollout(context: Context, rules: Rules | None = None) -> Rollout:
    selected_rules: Final = RULES if rules is None else rules
    return next((rule.rollout for rule in selected_rules if rule.matches(context)), Rollout.PYTHON_ONLY)


def decision(context: Context, rules: Rules | None = None) -> Decision:
    return _decision(rollout(context, rules))
