"""Ordered rollout policy for routes.

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


class Target(str, Enum):
    CHAT_COMPLETIONS = "chat_completions"
    MESSAGES = "messages"
    RESPONSES = "responses"


class Feature(str, Enum):
    SAFEGUARDS_CLASSIFIER = "safeguards_classifier"
    MOCK_RESPONSE = "mock_response"
    PROXY_MCP_TOOLS = "proxy_mcp_tools"
    CALLER_CLIENT = "caller_client"
    TIMEOUT_VARIANTS = "timeout_variants"
    DROP_MODIFY_PARAMS = "drop_modify_params"
    STREAMING = "streaming"
    PROMPT_MANAGEMENT = "prompt_management"
    TEXT_FORMAT = "text_format"
    RETRIES_FALLBACKS = "retries_fallbacks"
    DEPLOYMENT_RESPONSE_ID = "deployment_response_id"
    ENCODED_ITEM_IDS = "encoded_item_ids"
    PRE_REQUEST_HOOK = "pre_request_hook"
    WEBSEARCH_INTERCEPTION = "websearch_interception"
    MESSAGES_INTERCEPTOR = "messages_interceptor"
    AGENTIC_LOOP_HOOK = "agentic_loop_hook"
    RESPONSE_CACHE = "response_cache"
    SSL_SECURITY_LEVEL = "ssl_security_level"


@dataclass(frozen=True, slots=True)
class RouteContext:
    route: Route
    provider: str | None = None
    model: str | None = None
    target: Target | None = None
    features: frozenset[Feature] = frozenset()

    def axes(self) -> tuple[frozenset[object], ...]:
        return (frozenset({self.provider}), frozenset({self.model}), frozenset({self.target}), self.features)


@dataclass(frozen=True, slots=True)
class RouteRule:
    route: Route
    rollout: Rollout
    providers: frozenset[str] | None = None
    models: frozenset[str] | None = None
    targets: frozenset[Target] | None = None
    features: frozenset[Feature] | None = None

    def constraints(self) -> tuple[frozenset[object] | None, ...]:
        return (self.providers, self.models, self.targets, self.features)

    def matches(self, context: RouteContext) -> bool:
        return context.route is self.route and all(
            allowed is None or not allowed.isdisjoint(values)
            for allowed, values in zip(self.constraints(), context.axes(), strict=True)
        )


Rules: TypeAlias = tuple[RouteRule, ...]

COMMON_PYTHON_FEATURES: Final = frozenset(
    {
        Feature.MOCK_RESPONSE,
        Feature.PROXY_MCP_TOOLS,
        Feature.CALLER_CLIENT,
        Feature.TIMEOUT_VARIANTS,
        Feature.AGENTIC_LOOP_HOOK,
        Feature.SSL_SECURITY_LEVEL,
    }
)

MESSAGES_PYTHON_FEATURES: Final = COMMON_PYTHON_FEATURES | {
    Feature.SAFEGUARDS_CLASSIFIER,
    Feature.PRE_REQUEST_HOOK,
    Feature.WEBSEARCH_INTERCEPTION,
    Feature.MESSAGES_INTERCEPTOR,
    Feature.RESPONSE_CACHE,
}

RESPONSES_PYTHON_FEATURES: Final = COMMON_PYTHON_FEATURES | {
    Feature.DROP_MODIFY_PARAMS,
    Feature.STREAMING,
    Feature.PROMPT_MANAGEMENT,
    Feature.TEXT_FORMAT,
    Feature.RETRIES_FALLBACKS,
    Feature.DEPLOYMENT_RESPONSE_ID,
    Feature.ENCODED_ITEM_IDS,
}

RULES: Final[Rules] = (
    RouteRule(Route.CHAT_COMPLETIONS, Rollout.PYTHON_ONLY),
    RouteRule(Route.EMBEDDINGS, Rollout.PYTHON_ONLY),
    RouteRule(Route.OCR, Rollout.RUST_REQUIRED),
    RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY, targets=frozenset({Target.CHAT_COMPLETIONS, Target.RESPONSES})),
    RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY, features=MESSAGES_PYTHON_FEATURES),
    RouteRule(Route.MESSAGES, Rollout.RUST_OPT_IN, providers=frozenset({"anthropic"})),
    RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY),
    RouteRule(Route.RESPONSES, Rollout.PYTHON_ONLY, features=RESPONSES_PYTHON_FEATURES),
    RouteRule(Route.RESPONSES, Rollout.PYTHON_ONLY),
    RouteRule(Route.TOKEN_COUNTER, Rollout.PYTHON_ONLY),
    RouteRule(Route.TOKENIZER, Rollout.PYTHON_ONLY),
    RouteRule(Route.TRANSCRIPTION, Rollout.RUST_REQUIRED, providers=frozenset({"bedrock"})),
)


def rollout(context: RouteContext, rules: Rules | None = None) -> Rollout:
    selected_rules: Final = RULES if rules is None else rules
    return next((rule.rollout for rule in selected_rules if rule.matches(context)), Rollout.PYTHON_ONLY)


def decision(context: RouteContext, rules: Rules | None = None) -> Decision:
    return _decision(rollout(context, rules))


def may_leave_python(route: Route, rules: Rules) -> bool:
    own: Final = tuple(rule for rule in rules if rule.route is route)
    last: Final = next((i for i, rule in enumerate(own) if all(c is None for c in rule.constraints())), len(own))
    return any(_decision(rule.rollout) is not Decision.PYTHON for rule in own[: last + 1])
