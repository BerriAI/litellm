from __future__ import annotations

from typing import Final

import pytest

from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Context, Delivery, Route, Rule
from litellm.rust_bridge.configuration import Rollout


def test_every_route_has_an_explicit_default_rule() -> None:
    declared: Final = frozenset(
        rule.route for rule in catalog.RULES if rule.providers is None and rule.deliveries is None
    )
    assert declared == frozenset(Route)


@pytest.mark.parametrize(
    ("context", "expected"),
    (
        (Context(Route.OCR), Rollout.RUST_OPT_OUT),
        (Context(Route.OCR, provider="mistral", model="mistral-ocr-latest"), Rollout.RUST_OPT_OUT),
        (Context(Route.TRANSCRIPTION, provider="bedrock"), Rollout.RUST_REQUIRED),
        (Context(Route.TRANSCRIPTION, provider="openai"), Rollout.PYTHON_ONLY),
        (Context(Route.TRANSCRIPTION), Rollout.PYTHON_ONLY),
        (Context(Route.CHAT_COMPLETIONS, provider="anthropic"), Rollout.RUST_OPT_IN),
        (Context(Route.CHAT_COMPLETIONS, provider="bedrock"), Rollout.RUST_OPT_IN),
        (Context(Route.CHAT_COMPLETIONS, provider="anthropic", delivery=Delivery.STREAMING), Rollout.PYTHON_ONLY),
        (Context(Route.CHAT_COMPLETIONS, provider="openai"), Rollout.PYTHON_ONLY),
        (Context(Route.MESSAGES, provider="anthropic"), Rollout.RUST_OPT_IN),
        (Context(Route.MESSAGES, provider="azure_ai"), Rollout.RUST_OPT_IN),
        (Context(Route.MESSAGES, provider="bedrock"), Rollout.PYTHON_ONLY),
        (Context(Route.RESPONSES, provider="openai", delivery=Delivery.WEBSOCKET), Rollout.RUST_OPT_IN),
        (Context(Route.RESPONSES, provider="openai"), Rollout.PYTHON_ONLY),
        (Context(Route.RESPONSES, provider="azure", delivery=Delivery.WEBSOCKET), Rollout.PYTHON_ONLY),
        (Context(Route.EMBEDDING, provider="openai"), Rollout.PYTHON_ONLY),
    ),
)
def test_shipped_rules(context: Context, expected: Rollout) -> None:
    assert catalog.rollout(context) is expected


def test_first_matching_rule_wins() -> None:
    rules: Final = (
        Rule(Route.EMBEDDING, Rollout.RUST_REQUIRED, providers=frozenset({"openai"}), models=frozenset({"m"})),
        Rule(Route.EMBEDDING, Rollout.RUST_OPT_IN, providers=frozenset({"openai"})),
        Rule(Route.EMBEDDING, Rollout.PYTHON_ONLY),
    )

    assert catalog.rollout(Context(Route.EMBEDDING, provider="openai", model="m"), rules) is Rollout.RUST_REQUIRED
    assert catalog.rollout(Context(Route.EMBEDDING, provider="openai", model="other"), rules) is Rollout.RUST_OPT_IN
    assert catalog.rollout(Context(Route.EMBEDDING, provider="cohere", model="m"), rules) is Rollout.PYTHON_ONLY
    assert catalog.rollout(Context(Route.RERANK, provider="openai", model="m"), rules) is Rollout.PYTHON_ONLY
