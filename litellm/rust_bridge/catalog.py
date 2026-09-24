"""Ordered rollout policy for routes, cache backends, and secret managers.

The first matching rule wins; unmatched contexts stay on Python. Native
admission separately decides whether the selected implementation can execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Final, TypeAlias

from litellm.rust_bridge.configuration import Decision, Rollout
from litellm.rust_bridge.configuration import decision as _decision
from litellm.types.caching import LiteLLMCacheType
from litellm.types.secret_managers.main import KeyManagementSystem


class Route(str, Enum):
    CHAT_COMPLETIONS = "chat_completions"
    EMBEDDINGS = "embeddings"
    MESSAGES = "messages"
    RESPONSES = "responses"
    TRANSCRIPTION = "transcription"
    OCR = "ocr"
    TOKEN_COUNTER = "token_counter"
    TOKENIZER = "tokenizer"


class Delivery(Enum):
    COMPLETED = auto()
    STREAMING = auto()
    WEBSOCKET = auto()


@dataclass(frozen=True, slots=True)
class RouteContext:
    route: Route
    provider: str | None = None
    model: str | None = None
    delivery: Delivery = Delivery.COMPLETED


@dataclass(frozen=True, slots=True)
class RouteRule:
    route: Route
    rollout: Rollout
    providers: frozenset[str] | None = None
    models: frozenset[str] | None = None
    deliveries: frozenset[Delivery] | None = None

    def matches(self, context: Context) -> bool:
        return (
            isinstance(context, RouteContext)
            and context.route is self.route
            and (self.providers is None or context.provider in self.providers)
            and (self.models is None or context.model in self.models)
            and (self.deliveries is None or context.delivery in self.deliveries)
        )


@dataclass(frozen=True, slots=True)
class CacheContext:
    backend: str


@dataclass(frozen=True, slots=True)
class CacheRule:
    rollout: Rollout
    backends: frozenset[str] | None = None

    def matches(self, context: Context) -> bool:
        return isinstance(context, CacheContext) and (self.backends is None or context.backend in self.backends)


@dataclass(frozen=True, slots=True)
class SecretManagerContext:
    system: str


@dataclass(frozen=True, slots=True)
class SecretManagerRule:
    rollout: Rollout
    systems: frozenset[str] | None = None

    def matches(self, context: Context) -> bool:
        return isinstance(context, SecretManagerContext) and (self.systems is None or context.system in self.systems)


@dataclass(frozen=True, slots=True)
class LoggerContext:
    pass


@dataclass(frozen=True, slots=True)
class LoggerRule:
    rollout: Rollout

    def matches(self, context: Context) -> bool:
        return isinstance(context, LoggerContext)


Context: TypeAlias = RouteContext | CacheContext | SecretManagerContext | LoggerContext
Rule: TypeAlias = RouteRule | CacheRule | SecretManagerRule | LoggerRule
Rules: TypeAlias = tuple[Rule, ...]

RULES: Final[Rules] = (
    LoggerRule(Rollout.RUST_OPT_IN),
    RouteRule(Route.CHAT_COMPLETIONS, Rollout.PYTHON_ONLY),
    RouteRule(Route.EMBEDDINGS, Rollout.PYTHON_ONLY),
    RouteRule(Route.OCR, Rollout.RUST_REQUIRED, providers=frozenset({"aws_textract"})),
    RouteRule(Route.OCR, Rollout.RUST_OPT_OUT),
    RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY),
    RouteRule(Route.RESPONSES, Rollout.PYTHON_ONLY),
    RouteRule(Route.TOKEN_COUNTER, Rollout.PYTHON_ONLY),
    RouteRule(Route.TOKENIZER, Rollout.PYTHON_ONLY),
    RouteRule(Route.TRANSCRIPTION, Rollout.RUST_REQUIRED, providers=frozenset({"bedrock"})),
    CacheRule(Rollout.PYTHON_ONLY, backends=frozenset({LiteLLMCacheType.LOCAL})),
    CacheRule(Rollout.PYTHON_ONLY, backends=frozenset({LiteLLMCacheType.REDIS})),
    CacheRule(Rollout.PYTHON_ONLY, backends=frozenset({LiteLLMCacheType.REDIS_SEMANTIC})),
    CacheRule(Rollout.PYTHON_ONLY, backends=frozenset({LiteLLMCacheType.VALKEY_SEMANTIC})),
    CacheRule(Rollout.PYTHON_ONLY, backends=frozenset({LiteLLMCacheType.S3})),
    CacheRule(Rollout.PYTHON_ONLY, backends=frozenset({LiteLLMCacheType.DISK})),
    CacheRule(Rollout.PYTHON_ONLY, backends=frozenset({LiteLLMCacheType.QDRANT_SEMANTIC})),
    CacheRule(Rollout.PYTHON_ONLY, backends=frozenset({LiteLLMCacheType.AZURE_BLOB})),
    CacheRule(Rollout.PYTHON_ONLY, backends=frozenset({LiteLLMCacheType.GCS})),
    SecretManagerRule(Rollout.PYTHON_ONLY, systems=frozenset({KeyManagementSystem.GOOGLE_KMS.value})),
    SecretManagerRule(Rollout.PYTHON_ONLY, systems=frozenset({KeyManagementSystem.AZURE_KEY_VAULT.value})),
    SecretManagerRule(Rollout.PYTHON_ONLY, systems=frozenset({KeyManagementSystem.AWS_SECRET_MANAGER.value})),
    SecretManagerRule(Rollout.PYTHON_ONLY, systems=frozenset({KeyManagementSystem.GOOGLE_SECRET_MANAGER.value})),
    SecretManagerRule(Rollout.PYTHON_ONLY, systems=frozenset({KeyManagementSystem.HASHICORP_VAULT.value})),
    SecretManagerRule(Rollout.PYTHON_ONLY, systems=frozenset({KeyManagementSystem.CYBERARK.value})),
    SecretManagerRule(Rollout.PYTHON_ONLY, systems=frozenset({KeyManagementSystem.LOCAL.value})),
    SecretManagerRule(Rollout.PYTHON_ONLY, systems=frozenset({KeyManagementSystem.AWS_KMS.value})),
    SecretManagerRule(Rollout.PYTHON_ONLY, systems=frozenset({KeyManagementSystem.CUSTOM.value})),
)


def rollout(context: Context, rules: Rules | None = None) -> Rollout:
    selected_rules: Final = RULES if rules is None else rules
    return next((rule.rollout for rule in selected_rules if rule.matches(context)), Rollout.PYTHON_ONLY)


def decision(context: Context, rules: Rules | None = None) -> Decision:
    return _decision(rollout(context, rules))
