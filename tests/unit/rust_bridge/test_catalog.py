from __future__ import annotations

from collections.abc import Generator
from typing import Final

import pytest

from litellm.litellm_core_utils.model_latest_aliases import get_model_latest_aliases
from litellm.rust_bridge import catalog, configuration, detectors
from litellm.rust_bridge.catalog import (
    Feature,
    Route,
    RouteContext,
    RouteRule,
    Rules,
    Target,
)
from litellm.rust_bridge.configuration import Decision, Rollout
from litellm.rust_bridge.public_call import NativeCall


@pytest.mark.parametrize(
    "model",
    ("anthropic/claude-opus-latest", f"anthropic/{get_model_latest_aliases()['anthropic']['claude-opus-latest']}"),
)
def test_latest_alias_messages_stays_native_like_the_pinned_model(model: str) -> None:
    request: Final = NativeCall(args=(), kwargs={"model": model}, base={})
    context: Final = RouteContext(
        Route.MESSAGES,
        provider="anthropic",
        model=model,
        features=detectors.detected(detectors.COMMON, request),
    )
    rules: Final = (
        RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY, features=catalog.MESSAGES_PYTHON_FEATURES),
        RouteRule(Route.MESSAGES, Rollout.RUST_OPT_IN, providers=frozenset({"anthropic"})),
    )
    assert catalog.rollout(context, rules) is Rollout.RUST_OPT_IN


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
        (RouteContext(Route.OCR, features=frozenset({Feature.STREAMING})), Decision.PYTHON),
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
        RouteRule(Route.OCR, Rollout.PYTHON_ONLY, features=frozenset({Feature.STREAMING})),
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
    assert catalog.decision(RouteContext(Route.OCR, features=frozenset({Feature.STREAMING})), rules) is Decision.PYTHON


@pytest.mark.parametrize(
    ("context", "expected"),
    (
        (RouteContext(Route.MESSAGES, provider="bedrock", target=Target.MESSAGES), Decision.RUST_REQUIRED),
        (RouteContext(Route.MESSAGES, provider="bedrock", target=Target.CHAT_COMPLETIONS), Decision.PYTHON),
        (RouteContext(Route.MESSAGES, provider="anthropic", target=Target.MESSAGES), Decision.PYTHON),
        (
            RouteContext(
                Route.MESSAGES,
                provider="bedrock",
                target=Target.MESSAGES,
                features=frozenset({Feature.STREAMING}),
            ),
            Decision.PYTHON,
        ),
    ),
)
def test_a_provider_and_target_rule_admits_only_that_pair(context: RouteContext, expected: Decision) -> None:
    rules: Final[Rules] = (
        RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY, features=frozenset({Feature.STREAMING})),
        RouteRule(
            Route.MESSAGES,
            Rollout.RUST_REQUIRED,
            providers=frozenset({"bedrock"}),
            targets=frozenset({Target.MESSAGES}),
        ),
        RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY),
    )

    assert catalog.decision(context, rules) is expected


@pytest.mark.parametrize(
    ("model", "target", "expected"),
    (
        ("vertex_ai/claude-sonnet-5-5@default", Target.MESSAGES, Decision.RUST_REQUIRED),
        ("vertex_ai/claude-opus-5-5", Target.MESSAGES, Decision.RUST_REQUIRED),
        ("vertex_ai/gemini-2.5-pro", Target.CHAT_COMPLETIONS, Decision.PYTHON),
        ("vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas", Target.CHAT_COMPLETIONS, Decision.PYTHON),
        ("vertex_ai/mistral-large-2411", Target.CHAT_COMPLETIONS, Decision.PYTHON),
    ),
)
def test_only_claude_on_vertex_ai_messages_is_served_by_rust(model: str, target: Target, expected: Decision) -> None:
    from litellm.messages.dispatch import messages_target

    resolved_target: Final = messages_target(model, None, None, None)
    context: Final = RouteContext(Route.MESSAGES, provider="vertex_ai", model=model, target=resolved_target)

    assert resolved_target is target
    assert catalog.decision(context) is expected


@pytest.mark.parametrize(
    ("provider", "target", "expected"),
    (
        ("anthropic", Target.MESSAGES, Decision.RUST_REQUIRED),
        ("azure_ai", Target.MESSAGES, Decision.RUST_REQUIRED),
        ("vertex_ai", Target.MESSAGES, Decision.RUST_REQUIRED),
        ("anthropic", None, Decision.PYTHON),
        ("azure_ai", None, Decision.PYTHON),
        ("vertex_ai", None, Decision.PYTHON),
        ("azure_ai", Target.CHAT_COMPLETIONS, Decision.PYTHON),
        ("azure_ai", Target.RESPONSES, Decision.PYTHON),
        ("bedrock", Target.MESSAGES, Decision.RUST_REQUIRED),
        ("bedrock_mantle", Target.MESSAGES, Decision.RUST_REQUIRED),
        ("bedrock", None, Decision.PYTHON),
        ("bedrock_mantle", None, Decision.PYTHON),
        ("bedrock", Target.CHAT_COMPLETIONS, Decision.PYTHON),
        ("bedrock_mantle", Target.RESPONSES, Decision.PYTHON),
        ("bedrock", Target.MESSAGES_PASSTHROUGH, Decision.PYTHON),
        ("openai", Target.MESSAGES, Decision.PYTHON),
    ),
)
def test_native_messages_providers_require_rust_only_for_a_native_messages_target(
    provider: str, target: Target | None, expected: Decision
) -> None:
    context: Final = RouteContext(Route.MESSAGES, provider=provider, target=target)

    assert catalog.decision(context) is expected


def test_vertex_ai_messages_passthrough_stays_on_python() -> None:
    context: Final = RouteContext(Route.MESSAGES, provider="vertex_ai", target=Target.MESSAGES_PASSTHROUGH)

    assert catalog.decision(context) is Decision.PYTHON


def test_vertex_ai_claude_messages_has_no_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    configuration.rust(False)
    context: Final = RouteContext(Route.MESSAGES, provider="vertex_ai", target=Target.MESSAGES)

    assert catalog.rollout(context) is Rollout.RUST_REQUIRED
    assert catalog.decision(context) is Decision.RUST_REQUIRED
