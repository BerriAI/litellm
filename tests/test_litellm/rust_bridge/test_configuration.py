from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import pytest

from litellm.rust_bridge import configuration


@pytest.fixture(autouse=True)
def _isolated_configuration(  # pyright: ignore[reportUnusedFunction]  # pytest discovers fixtures dynamically
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None]:
    configuration.reset_rust_configuration()
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    yield
    configuration.reset_rust_configuration()


Rollout: Final = configuration.Rollout
Decision: Final = configuration.Decision


@pytest.mark.parametrize(
    ("rollout", "process", "environment", "expected"),
    (
        (Rollout.PYTHON_ONLY, True, True, Decision.PYTHON),
        (Rollout.RUST_REQUIRED, False, False, Decision.RUST_REQUIRED),
        (Rollout.RUST_OPT_IN, None, None, Decision.PYTHON),
        (Rollout.RUST_OPT_IN, None, True, Decision.RUST_WITH_FALLBACK),
        (Rollout.RUST_OPT_IN, True, False, Decision.RUST_WITH_FALLBACK),
        (Rollout.RUST_OPT_IN, False, True, Decision.PYTHON),
        (Rollout.RUST_OPT_OUT, None, None, Decision.RUST_WITH_FALLBACK),
        (Rollout.RUST_OPT_OUT, None, False, Decision.PYTHON),
        (Rollout.RUST_OPT_OUT, False, True, Decision.PYTHON),
        (Rollout.RUST_OPT_OUT, True, False, Decision.RUST_WITH_FALLBACK),
    ),
)
def test_decide_precedence(
    rollout: configuration.Rollout,
    process: bool | None,
    environment: bool | None,
    expected: configuration.Decision,
) -> None:
    assert configuration.decide(rollout, process_override=process, environment_override=environment) is expected


def test_release_default_keeps_opt_in_routes_on_python() -> None:
    assert configuration.decision(Rollout.RUST_OPT_IN) is Decision.PYTHON
    assert configuration.decision(Rollout.RUST_OPT_OUT) is Decision.RUST_WITH_FALLBACK
    assert configuration.rust_enabled() is False


@pytest.mark.parametrize("process", (None, False, True))
@pytest.mark.parametrize("environment", (None, "0", "1", "off"))
def test_opt_out_route_configuration(
    monkeypatch: pytest.MonkeyPatch, process: bool | None, environment: str | None
) -> None:
    if environment is not None:
        monkeypatch.setenv("LITELLM_RUST", environment)
    if process is not None:
        configuration.rust(process)

    expected: Final = (
        Decision.RUST_WITH_FALLBACK
        if process is True or (process is None and environment not in frozenset({"0", "off"}))
        else Decision.PYTHON
    )
    assert configuration.decision(Rollout.RUST_OPT_OUT) is expected


def test_process_override_wins_over_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    configuration.rust(True)

    assert configuration.rust_enabled() is True


def test_global_environment_accepts_explicit_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "off")

    assert configuration.rust_enabled() is False


@pytest.mark.parametrize("value", ("", " ", "sometimes", "2"))
def test_invalid_environment_value_disables_rust(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LITELLM_RUST", value)

    assert configuration.rust_enabled() is False


def test_process_override_and_reset_apply_to_existing_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "1")

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(configuration.rust_enabled).result() is True
        configuration.rust(False)
        assert executor.submit(configuration.rust_enabled).result() is False
        configuration.reset_rust_configuration()
        assert executor.submit(configuration.rust_enabled).result() is True


def test_explicit_override_precedes_invalid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "sometimes")

    configuration.rust(True)
    assert configuration.rust_enabled() is True


@pytest.mark.parametrize(("value", "expected"), (("1", "True"), ("0", "False")))
def test_environment_controls_startup(value: str, expected: str) -> None:
    environment: Final = {**os.environ, "LITELLM_RUST": value}
    result: Final = subprocess.run(
        (
            sys.executable,
            "-c",
            "from litellm.rust_bridge.configuration import rust_enabled; print(rust_enabled())",
        ),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.stdout.strip() == expected
