from __future__ import annotations

from collections.abc import Generator
from types import MappingProxyType
from typing import Final

import pytest

from litellm.rust_bridge import catalog, configuration
from litellm.rust_bridge.catalog import Decision, Python, Route, RouteContext, Rust

SWITCHED_OFF: Final = Python("Rust is switched off")


def only_openai(context: RouteContext) -> Decision:
    return catalog.optional() if context.provider == "openai" else Python("test keeps other providers on Python")


POLICIES: Final = MappingProxyType(
    {
        Route.CHAT_COMPLETIONS: catalog.required,
        Route.EMBEDDINGS: only_openai,
    }
)


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    yield
    configuration.reset_rust_configuration()


@pytest.mark.parametrize("enabled", (False, True), ids=("rust-off", "rust-on"))
@pytest.mark.parametrize(
    ("context", "when_off", "when_on"),
    (
        pytest.param(
            RouteContext(Route.CHAT_COMPLETIONS, provider="anything", model="m"),
            Rust(required=True),
            Rust(required=True),
            id="required-policy",
        ),
        pytest.param(RouteContext(Route.EMBEDDINGS, provider="openai"), SWITCHED_OFF, Rust(), id="optional-policy"),
        pytest.param(
            RouteContext(Route.EMBEDDINGS, provider="cohere"),
            Python("test keeps other providers on Python"),
            Python("test keeps other providers on Python"),
            id="policy-reads-the-context",
        ),
        pytest.param(
            RouteContext(Route.OCR, provider="mistral"),
            Python("ocr is not ported"),
            Python("ocr is not ported"),
            id="no-policy",
        ),
    ),
)
def test_decide_runs_the_route_policy(
    enabled: bool, context: RouteContext, when_off: Decision, when_on: Decision
) -> None:
    configuration.rust(enabled)

    assert catalog.decide(context, policies=POLICIES) == (when_on if enabled else when_off)


def test_optional_and_logger_follow_the_global_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    assert catalog.optional() == SWITCHED_OFF
    assert catalog.logger() == SWITCHED_OFF

    configuration.rust(True)
    assert catalog.optional() == Rust()
    assert catalog.logger() == Rust()

    monkeypatch.setenv("LITELLM_RUST", "0")
    assert catalog.optional() == SWITCHED_OFF
    assert catalog.logger() == SWITCHED_OFF
