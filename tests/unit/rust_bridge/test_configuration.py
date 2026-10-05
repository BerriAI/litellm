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
        (Rollout.RUST_OPT_IN, True, None, Decision.RUST_WITH_FALLBACK),
        (Rollout.RUST_OPT_IN, True, False, Decision.PYTHON),
        (Rollout.RUST_OPT_IN, False, True, Decision.RUST_WITH_FALLBACK),
        (Rollout.RUST_OPT_OUT, None, None, Decision.RUST_WITH_FALLBACK),
        (Rollout.RUST_OPT_OUT, None, False, Decision.PYTHON),
        (Rollout.RUST_OPT_OUT, False, None, Decision.PYTHON),
        (Rollout.RUST_OPT_OUT, False, True, Decision.RUST_WITH_FALLBACK),
        (Rollout.RUST_OPT_OUT, True, False, Decision.PYTHON),
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
        if environment == "1" or (environment is None and process is not False)
        else Decision.PYTHON
    )
    assert configuration.decision(Rollout.RUST_OPT_OUT) is expected


@pytest.mark.parametrize(
    ("environment", "process", "expected"),
    (
        *((value, True, False) for value in ("0", "false", "False", "no", "off", "f", "n", " 0 ")),
        *((value, False, True) for value in ("1", "true", "TRUE", "yes", "on", "t", "y", " 1 ")),
    ),
)
def test_environment_wins_over_process_override(
    monkeypatch: pytest.MonkeyPatch, environment: str, process: bool, expected: bool
) -> None:
    monkeypatch.setenv("LITELLM_RUST", environment)
    configuration.rust(process)

    assert configuration.rust_enabled() is expected


def test_process_override_applies_when_environment_is_unset() -> None:
    configuration.rust(True)

    assert configuration.rust_enabled() is True


@pytest.mark.parametrize("value", ("", " ", "sometimes", "2"))
def test_invalid_environment_value_is_ignored(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LITELLM_RUST", value)

    assert configuration.rust_enabled() is False
    assert configuration.decision(Rollout.RUST_OPT_OUT) is Decision.RUST_WITH_FALLBACK
    configuration.rust(True)
    assert configuration.rust_enabled() is True


def test_process_override_and_reset_apply_to_existing_threads() -> None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(configuration.rust_enabled).result() is False
        configuration.rust(True)
        assert executor.submit(configuration.rust_enabled).result() is True
        configuration.reset_rust_configuration()
        assert executor.submit(configuration.rust_enabled).result() is False


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
