from __future__ import annotations

from collections.abc import Generator
from typing import Final

import pytest

from litellm.rust_bridge import catalog, configuration
from litellm.rust_bridge.catalog import Python, Route, RouteContext, Rust


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    yield
    configuration.reset_rust_configuration()


@pytest.mark.parametrize("route", tuple(Route))
@pytest.mark.parametrize("provider", (None, "bedrock", "mistral", "anthropic", "openai", "azure_ai", "unknown"))
@pytest.mark.parametrize("process", (None, False, True))
@pytest.mark.parametrize("environment", (None, "0", "1"))
def test_shipped_decisions(
    monkeypatch: pytest.MonkeyPatch,
    route: Route,
    provider: str | None,
    process: bool | None,
    environment: str | None,
) -> None:
    configuration.rust(process)
    if environment is not None:
        monkeypatch.setenv("LITELLM_RUST", environment)
    decision: Final = catalog.decide(RouteContext(route, provider=provider, model="test-model"))

    if route is Route.OCR or (route is Route.TRANSCRIPTION and provider == "bedrock"):
        assert decision == Rust(required=True)
    elif route is Route.MESSAGES and provider == "anthropic":
        opted_in: Final = environment == "1" or (environment is None and process is True)
        assert decision == (Rust() if opted_in else Python("Rust is switched off"))
    else:
        assert isinstance(decision, Python)


@pytest.mark.parametrize("process", (None, False, True))
@pytest.mark.parametrize("environment", (None, "0", "1"))
def test_unported_routes_stay_on_python_whatever_the_switch(
    monkeypatch: pytest.MonkeyPatch, process: bool | None, environment: str | None
) -> None:
    configuration.rust(process)
    if environment is not None:
        monkeypatch.setenv("LITELLM_RUST", environment)

    assert catalog.decide(RouteContext(Route.CHAT_COMPLETIONS, provider="anthropic")) == Python(
        "chat_completions is not ported"
    )
    assert catalog.decide(RouteContext(Route.MESSAGES, provider="openai")) == Python(
        "only Anthropic Messages is ported"
    )
    assert catalog.decide(RouteContext(Route.TRANSCRIPTION, provider="openai")) == Python(
        "only Bedrock transcription is ported"
    )


def test_logger_follows_the_global_switch() -> None:
    assert catalog.logger() == Python("Rust is switched off")
    configuration.rust(True)
    assert catalog.logger() == Rust()


@pytest.mark.parametrize("process", (None, False, True))
@pytest.mark.parametrize("environment", (None, "0", "1"))
def test_ocr_has_no_python_path_to_opt_out_to(
    monkeypatch: pytest.MonkeyPatch, process: bool | None, environment: str | None
) -> None:
    configuration.rust(process)
    if environment is not None:
        monkeypatch.setenv("LITELLM_RUST", environment)

    assert catalog.decide(RouteContext(Route.OCR, model="m")) == Rust(required=True)
    assert catalog.decide(RouteContext(Route.OCR, provider="aws_textract", model="m")) == Rust(required=True)


@pytest.mark.parametrize(
    ("process", "environment", "expected"),
    (
        (None, None, Python("Rust is switched off")),
        (True, None, Rust()),
        (False, None, Python("Rust is switched off")),
        (False, "1", Rust()),
        (True, "0", Python("Rust is switched off")),
    ),
)
def test_optional_reads_environment_then_process_override(
    monkeypatch: pytest.MonkeyPatch, process: bool | None, environment: str | None, expected: catalog.Decision
) -> None:
    configuration.rust(process)
    if environment is not None:
        monkeypatch.setenv("LITELLM_RUST", environment)

    assert catalog.optional() == expected
