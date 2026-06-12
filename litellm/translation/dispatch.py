"""The one v1/v2 fork.

A pure decision: a request stays on v1 until its provider is opted in, takes
the same-family fast path when the inbound schema already speaks the provider's
wire format (so the IR semantic re-encode is skipped), and otherwise goes
through the IR. Enabling any body-touching feature (param normalization,
guardrails, semantic cache) sets ``body_touching`` and forces the IR path even
for a same-family pair, because the fast path forwards the message payload as-is.
"""

from __future__ import annotations

from typing import Literal

from expression import case, tag, tagged_union

from .ir import UNIT, Unit

InboundSchema = Literal[
    "openai_chat", "anthropic_messages", "google_genai", "responses", "completions"
]

Provider = Literal[
    "anthropic",
    "bedrock_converse",
    "bedrock_invoke",
    "vertex_ai",
    "azure",
    "azure_ai",
    "azure_ai_anthropic",
    "openai_compat",
    "gemini",
    "vertex_anthropic",
    "xai",
]

_SAME_FAMILY: frozenset[tuple[InboundSchema, Provider]] = frozenset(
    {
        ("anthropic_messages", "anthropic"),
        ("anthropic_messages", "bedrock_invoke"),
        ("anthropic_messages", "vertex_ai"),
        ("anthropic_messages", "vertex_anthropic"),
        ("openai_chat", "openai_compat"),
        # xai is NOT same-family despite speaking openai-chat: v1's transform
        # touches the body (tools strict strip, non-user message name strip),
        # so a verbatim fast-path forward would diverge from v1.
    }
)


@tagged_union(frozen=True)
class Route:
    tag: Literal["v1", "fast_path", "v2"] = tag()

    v1: Unit = case()
    fast_path: Provider = case()
    v2: Provider = case()

    @staticmethod
    def of_v1() -> Route:
        return Route(v1=UNIT)

    @staticmethod
    def of_fast_path(provider: Provider) -> Route:
        return Route(fast_path=provider)

    @staticmethod
    def of_v2(provider: Provider) -> Route:
        return Route(v2=provider)


def route(
    *,
    schema: InboundSchema,
    provider: Provider,
    enabled_providers: frozenset[Provider],
    body_touching: bool,
) -> Route:
    if provider not in enabled_providers:
        return Route.of_v1()
    if (schema, provider) in _SAME_FAMILY and not body_touching:
        return Route.of_fast_path(provider)
    return Route.of_v2(provider)
