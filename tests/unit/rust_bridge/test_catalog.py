from __future__ import annotations

from collections.abc import Generator
from typing import Final

import pytest

from litellm.rust_bridge import catalog, configuration
from litellm.rust_bridge.catalog import (
    CacheContext,
    CacheRule,
    Context,
    Delivery,
    LoggerContext,
    Route,
    RouteContext,
    RouteRule,
    Rules,
    SecretManagerContext,
    SecretManagerRule,
)
from litellm.rust_bridge.configuration import Decision, Rollout
from litellm.types.caching import LiteLLMCacheType
from litellm.types.secret_managers.main import KeyManagementSystem


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
    context: Final = RouteContext(route, provider=provider, model="test-model", delivery=delivery)

    if route is Route.OCR or (route is Route.TRANSCRIPTION and provider == "bedrock"):
        assert catalog.rollout(context) is Rollout.RUST_REQUIRED
        assert catalog.decision(context) is Decision.RUST_REQUIRED
    elif route is Route.MESSAGES and provider == "anthropic":
        assert catalog.rollout(context) is Rollout.RUST_OPT_IN
        opted_in: Final = environment == "1" or (environment is None and process is True)
        assert catalog.decision(context) is (Decision.RUST_WITH_FALLBACK if opted_in else Decision.PYTHON)
    else:
        assert catalog.rollout(context) is Rollout.PYTHON_ONLY
        assert catalog.decision(context) is Decision.PYTHON


@pytest.mark.parametrize("route", tuple(Route))
def test_missing_rule_stays_on_python_even_when_rust_is_enabled(monkeypatch: pytest.MonkeyPatch, route: Route) -> None:
    configuration.rust(True)
    monkeypatch.setenv("LITELLM_RUST", "1")

    assert catalog.rollout(RouteContext(route), rules=()) is Rollout.PYTHON_ONLY
    assert catalog.decision(RouteContext(route), rules=()) is Decision.PYTHON


@pytest.mark.parametrize(
    "context",
    (
        *(CacheContext(backend.value) for backend in LiteLLMCacheType),
        *(SecretManagerContext(system.value) for system in KeyManagementSystem),
        CacheContext("custom"),
        SecretManagerContext("unknown"),
    ),
)
def test_backend_rollouts_stay_on_python_when_global_rust_is_enabled(
    monkeypatch: pytest.MonkeyPatch, context: Context
) -> None:
    configuration.rust(True)
    monkeypatch.setenv("LITELLM_RUST", "1")

    assert catalog.rollout(context) is Rollout.PYTHON_ONLY
    assert catalog.decision(context) is Decision.PYTHON


def test_logger_rollout_obeys_the_global_switch() -> None:
    assert catalog.rollout(LoggerContext()) is Rollout.RUST_OPT_IN
    assert catalog.decision(LoggerContext()) is Decision.PYTHON
    configuration.rust(True)
    assert catalog.decision(LoggerContext()) is Decision.RUST_WITH_FALLBACK


def test_response_cache_rules_select_the_whole_backend_runtime() -> None:
    rules: Final = (
        CacheRule(Rollout.RUST_REQUIRED, backends=frozenset({"local"})),
        CacheRule(Rollout.PYTHON_ONLY),
    )

    assert catalog.decision(CacheContext(backend="local"), rules) is Decision.RUST_REQUIRED
    assert catalog.decision(CacheContext(backend="redis"), rules) is Decision.PYTHON


@pytest.mark.parametrize(
    ("context", "expected"),
    (
        (
            RouteContext(Route.RESPONSES, provider="openai", model="m", delivery=Delivery.WEBSOCKET),
            Decision.RUST_REQUIRED,
        ),
        (RouteContext(Route.RESPONSES, provider="openai", model="m"), Decision.PYTHON),
        (RouteContext(Route.RESPONSES, provider="openai", model="m", delivery=Delivery.STREAMING), Decision.PYTHON),
        (RouteContext(Route.RESPONSES, provider="openai", model="other", delivery=Delivery.WEBSOCKET), Decision.PYTHON),
        (RouteContext(Route.RESPONSES, provider="anthropic", model="m", delivery=Delivery.WEBSOCKET), Decision.PYTHON),
        (RouteContext(Route.MESSAGES, provider="openai", model="m", delivery=Delivery.WEBSOCKET), Decision.PYTHON),
    ),
)
def test_first_matching_rule_respects_every_constraint(context: RouteContext, expected: Decision) -> None:
    rules: Final = (
        RouteRule(
            Route.RESPONSES,
            Rollout.RUST_REQUIRED,
            providers=frozenset({"openai"}),
            models=frozenset({"m"}),
            deliveries=frozenset({Delivery.WEBSOCKET}),
        ),
        RouteRule(Route.RESPONSES, Rollout.PYTHON_ONLY),
    )

    assert catalog.decision(context, rules) is expected


@pytest.mark.parametrize("process", (None, False, True))
@pytest.mark.parametrize("environment", (None, "0", "1"))
def test_ocr_has_no_python_path_to_opt_out_to(
    monkeypatch: pytest.MonkeyPatch, process: bool | None, environment: str | None
) -> None:
    configuration.rust(process)
    if environment is not None:
        monkeypatch.setenv("LITELLM_RUST", environment)

    assert catalog.decision(RouteContext(Route.OCR, model="m")) is Decision.RUST_REQUIRED
    assert catalog.decision(RouteContext(Route.OCR, provider="aws_textract", model="m")) is Decision.RUST_REQUIRED


@pytest.mark.parametrize(
    ("context", "expected"),
    (
        (RouteContext(Route.OCR, provider="local"), Decision.RUST_REQUIRED),
        (RouteContext(Route.OCR, provider="other"), Decision.PYTHON),
        (RouteContext(Route.MESSAGES, provider="local"), Decision.PYTHON),
        (CacheContext("local"), Decision.RUST_WITH_FALLBACK),
        (CacheContext("other"), Decision.PYTHON),
        (SecretManagerContext("local"), Decision.PYTHON),
        (SecretManagerContext("other"), Decision.RUST_REQUIRED),
    ),
)
def test_mixed_rules_select_only_the_matching_domain(context: Context, expected: Decision) -> None:
    rules: Final[Rules] = (
        CacheRule(Rollout.RUST_OPT_OUT, backends=frozenset({"local"})),
        CacheRule(Rollout.PYTHON_ONLY),
        SecretManagerRule(Rollout.PYTHON_ONLY, systems=frozenset({"local"})),
        SecretManagerRule(Rollout.RUST_REQUIRED),
        RouteRule(Route.OCR, Rollout.RUST_REQUIRED, providers=frozenset({"local"})),
        RouteRule(Route.OCR, Rollout.PYTHON_ONLY),
    )

    assert catalog.decision(context, rules) is expected


@pytest.mark.parametrize("context", (RouteContext(Route.OCR), CacheContext("local"), SecretManagerContext("local")))
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
def test_all_domains_share_rollout_switches_and_first_match(
    monkeypatch: pytest.MonkeyPatch,
    context: Context,
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
        CacheRule(rollout),
        SecretManagerRule(rollout),
        RouteRule(Route.OCR, Rollout.RUST_REQUIRED),
        CacheRule(Rollout.RUST_REQUIRED),
        SecretManagerRule(Rollout.RUST_REQUIRED),
    )

    assert catalog.decision(context, rules) is expected
    assert catalog.decision(context, ()) is Decision.PYTHON


@pytest.mark.parametrize("context", (RouteContext(Route.OCR), CacheContext("local"), SecretManagerContext("local")))
def test_empty_constraints_match_nothing(context: Context) -> None:
    rules: Final[Rules] = (
        RouteRule(Route.OCR, Rollout.RUST_REQUIRED, providers=frozenset()),
        CacheRule(Rollout.RUST_REQUIRED, backends=frozenset()),
        SecretManagerRule(Rollout.RUST_REQUIRED, systems=frozenset()),
    )

    assert catalog.decision(context, rules) is Decision.PYTHON
