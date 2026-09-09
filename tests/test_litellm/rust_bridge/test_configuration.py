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


@pytest.mark.parametrize(
    ("process", "environment", "release_default", "expected"),
    (
        (False, True, True, False),
        (True, False, False, True),
        (None, False, True, False),
        (None, True, False, True),
        (None, None, False, False),
        (None, None, True, True),
    ),
)
def test_resolution_precedence(
    process: bool | None,
    environment: bool | None,
    release_default: bool,
    expected: bool,
) -> None:
    assert (
        configuration.resolve_rust_enabled(
            process_override=process,
            environment_override=environment,
            release_default=release_default,
        )
        is expected
    )


def test_release_default_remains_disabled() -> None:
    assert configuration.DEFAULT_RUST_ENABLED is False
    assert configuration.rust_enabled() is False


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
