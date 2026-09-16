from __future__ import annotations

from collections.abc import Generator
from typing import Final

import pytest

from litellm.rust_bridge import catalog, configuration
from litellm.rust_bridge.catalog import Context, Delivery, Route, Rule
from litellm.rust_bridge.configuration import Decision, Rollout


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    yield
    configuration.reset_rust_configuration()


@pytest.mark.parametrize("route", tuple(Route))
@pytest.mark.parametrize("provider", (None, "bedrock", "mistral", "anthropic", "openai", "azure_ai", "unknown"))
@pytest.mark.parametrize("delivery", tuple(Delivery))
@pytest.mark.parametrize("process", (None, False, True))
@pytest.mark.parametrize("environment", (None, "0", "1"))
def test_shipped_decisions(
    monkeypatch: pytest.MonkeyPatch,
    route: Route,
    provider: str | None,
    delivery: Delivery,
    process: bool | None,
    environment: str | None,
) -> None:
    configuration.rust(process)
    if environment is not None:
        monkeypatch.setenv("LITELLM_RUST", environment)
    context: Final = Context(route, provider=provider, model="test-model", delivery=delivery)

    if route is Route.OCR:
        enabled: Final = environment == "1" if environment is not None else process is not False
        assert catalog.rollout(context) is Rollout.RUST_OPT_OUT
        assert catalog.decision(context) is (Decision.RUST_WITH_FALLBACK if enabled else Decision.PYTHON)
    elif route is Route.TRANSCRIPTION and provider == "bedrock":
        assert catalog.rollout(context) is Rollout.RUST_REQUIRED
        assert catalog.decision(context) is Decision.RUST_REQUIRED
    else:
        assert catalog.rollout(context) is Rollout.PYTHON_ONLY
        assert catalog.decision(context) is Decision.PYTHON


@pytest.mark.parametrize("route", tuple(Route))
def test_missing_rule_stays_on_python_even_when_rust_is_enabled(monkeypatch: pytest.MonkeyPatch, route: Route) -> None:
    configuration.rust(True)
    monkeypatch.setenv("LITELLM_RUST", "1")

    assert catalog.rollout(Context(route), rules=()) is Rollout.PYTHON_ONLY
    assert catalog.decision(Context(route), rules=()) is Decision.PYTHON


@pytest.mark.parametrize(
    ("context", "expected"),
    (
        (Context(Route.RESPONSES, provider="openai", model="m", delivery=Delivery.WEBSOCKET), Decision.RUST_REQUIRED),
        (Context(Route.RESPONSES, provider="openai", model="m"), Decision.PYTHON),
        (Context(Route.RESPONSES, provider="openai", model="m", delivery=Delivery.STREAMING), Decision.PYTHON),
        (Context(Route.RESPONSES, provider="openai", model="other", delivery=Delivery.WEBSOCKET), Decision.PYTHON),
        (Context(Route.RESPONSES, provider="anthropic", model="m", delivery=Delivery.WEBSOCKET), Decision.PYTHON),
        (Context(Route.MESSAGES, provider="openai", model="m", delivery=Delivery.WEBSOCKET), Decision.PYTHON),
    ),
)
def test_first_matching_rule_respects_every_constraint(context: Context, expected: Decision) -> None:
    rules: Final = (
        Rule(
            Route.RESPONSES,
            Rollout.RUST_REQUIRED,
            providers=frozenset({"openai"}),
            models=frozenset({"m"}),
            deliveries=frozenset({Delivery.WEBSOCKET}),
        ),
        Rule(Route.RESPONSES, Rollout.PYTHON_ONLY),
    )

    assert catalog.decision(context, rules) is expected
