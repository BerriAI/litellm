from __future__ import annotations

from collections.abc import Generator
from typing import Final

import pytest

from litellm.rust_bridge import catalog, configuration
from litellm.rust_bridge.catalog import (
    Feature,
    Route,
    RouteContext,
    RouteRule,
    Rules,
    Target,
)
from litellm.rust_bridge.configuration import Decision, Rollout


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    yield
    configuration.reset_rust_configuration()


@pytest.mark.parametrize("route", tuple(Route))
def test_missing_rule_stays_on_python_even_when_rust_is_enabled(monkeypatch: pytest.MonkeyPatch, route: Route) -> None:
    configuration.rust(True)
    monkeypatch.setenv("LITELLM_RUST", "1")

    assert catalog.rollout(RouteContext(route), rules=()) is Rollout.PYTHON_ONLY
    assert catalog.decision(RouteContext(route), rules=()) is Decision.PYTHON


@pytest.mark.parametrize(
    ("context", "expected"),
    (
        (
            RouteContext(Route.RESPONSES, provider="openai", model="m"),
            Decision.RUST_REQUIRED,
        ),
        (RouteContext(Route.RESPONSES, provider="openai", model="other"), Decision.PYTHON),
        (RouteContext(Route.RESPONSES, provider="anthropic", model="m"), Decision.PYTHON),
        (RouteContext(Route.MESSAGES, provider="openai", model="m"), Decision.PYTHON),
        (RouteContext(Route.MESSAGES, target=Target.MESSAGES), Decision.RUST_REQUIRED),
        (RouteContext(Route.MESSAGES, target=Target.CHAT_COMPLETIONS), Decision.PYTHON),
        (RouteContext(Route.MESSAGES), Decision.PYTHON),
        (RouteContext(Route.OCR, features=frozenset({Feature.SAFEGUARDS_CLASSIFIER})), Decision.PYTHON),
        (RouteContext(Route.OCR), Decision.RUST_REQUIRED),
    ),
)
def test_first_matching_rule_respects_every_constraint(context: RouteContext, expected: Decision) -> None:
    rules: Final = (
        RouteRule(
            Route.RESPONSES,
            Rollout.RUST_REQUIRED,
            providers=frozenset({"openai"}),
            models=frozenset({"m"}),
        ),
        RouteRule(Route.RESPONSES, Rollout.PYTHON_ONLY),
        RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED, targets=frozenset({Target.MESSAGES})),
        RouteRule(Route.OCR, Rollout.PYTHON_ONLY, features=frozenset({Feature.SAFEGUARDS_CLASSIFIER})),
        RouteRule(Route.OCR, Rollout.RUST_REQUIRED),
    )

    assert catalog.decision(context, rules) is expected


@pytest.mark.parametrize(
    ("context", "expected"),
    (
        (RouteContext(Route.OCR, provider="local"), Decision.RUST_REQUIRED),
        (RouteContext(Route.OCR, provider="other"), Decision.PYTHON),
        (RouteContext(Route.MESSAGES, provider="local"), Decision.PYTHON),
    ),
)
def test_rules_select_only_the_matching_route(context: RouteContext, expected: Decision) -> None:
    rules: Final[Rules] = (
        RouteRule(Route.OCR, Rollout.RUST_REQUIRED, providers=frozenset({"local"})),
        RouteRule(Route.OCR, Rollout.PYTHON_ONLY),
    )

    assert catalog.decision(context, rules) is expected


@pytest.mark.parametrize(
    ("rollout", "process", "environment", "expected"),
    (
        (Rollout.PYTHON_ONLY, True, "1", Decision.PYTHON),
        (Rollout.RUST_REQUIRED, False, "0", Decision.RUST_REQUIRED),
        (Rollout.RUST_OPT_IN, None, None, Decision.PYTHON),
        (Rollout.RUST_OPT_OUT, None, None, Decision.RUST_WITH_FALLBACK),
        (Rollout.RUST_OPT_IN, True, None, Decision.RUST_WITH_FALLBACK),
        (Rollout.RUST_OPT_OUT, False, None, Decision.PYTHON),
        (Rollout.RUST_OPT_IN, False, "1", Decision.RUST_WITH_FALLBACK),
        (Rollout.RUST_OPT_OUT, True, "0", Decision.PYTHON),
    ),
)
def test_rollout_switches_and_first_match(
    monkeypatch: pytest.MonkeyPatch,
    rollout: Rollout,
    process: bool | None,
    environment: str | None,
    expected: Decision,
) -> None:
    configuration.rust(process)
    if environment is not None:
        monkeypatch.setenv("LITELLM_RUST", environment)
    rules: Final[Rules] = (
        RouteRule(Route.OCR, rollout),
        RouteRule(Route.OCR, Rollout.RUST_REQUIRED),
    )

    assert catalog.decision(RouteContext(Route.OCR), rules) is expected
    assert catalog.decision(RouteContext(Route.OCR), ()) is Decision.PYTHON


def test_empty_constraints_match_nothing() -> None:
    rules: Final[Rules] = (
        RouteRule(Route.OCR, Rollout.RUST_REQUIRED, providers=frozenset()),
        RouteRule(Route.OCR, Rollout.RUST_REQUIRED, models=frozenset()),
        RouteRule(Route.OCR, Rollout.RUST_REQUIRED, features=frozenset()),
    )

    assert catalog.decision(RouteContext(Route.OCR), rules) is Decision.PYTHON
    assert catalog.decision(
        RouteContext(Route.OCR, features=frozenset({Feature.SAFEGUARDS_CLASSIFIER})), rules
    ) is Decision.PYTHON
