from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import pytest

from litellm.rust_bridge import configuration
from litellm.rust_bridge import ocr as rust_ocr


@pytest.fixture(autouse=True)
def _isolated_configuration(  # pyright: ignore[reportUnusedFunction]  # pytest discovers fixtures dynamically
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None]:
    configuration.reset_rust_configuration()
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    monkeypatch.delenv("LITELLM_USE_RUST_OCR", raising=False)
    rust_ocr.set_rust_ocr(ocr=None, aocr=None)
    yield
    configuration.reset_rust_configuration()
    rust_ocr.set_rust_ocr(ocr=None, aocr=None)


@pytest.mark.parametrize(
    ("request_override", "process", "environment", "legacy_environment", "release_default", "expected"),
    (
        (False, True, True, True, True, False),
        (True, False, False, False, False, True),
        (None, False, True, True, True, False),
        (None, True, False, False, False, True),
        (None, None, False, True, True, False),
        (None, None, True, False, False, True),
        (None, None, None, False, True, False),
        (None, None, None, True, False, True),
        (None, None, None, None, False, False),
        (None, None, None, None, True, True),
    ),
)
def test_resolution_precedence(
    request_override: bool | None,
    process: bool | None,
    environment: bool | None,
    legacy_environment: bool | None,
    release_default: bool,
    expected: bool,
) -> None:
    assert (
        configuration.resolve_rust_enabled(
            request_override=request_override,
            process_override=process,
            environment_override=environment,
            legacy_environment_override=legacy_environment,
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
    assert configuration.rust_enabled(request_override=False) is False


def test_global_environment_accepts_explicit_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "off")

    assert configuration.rust_enabled() is False


@pytest.mark.parametrize("value", ("", " ", "sometimes", "2"))
def test_invalid_environment_value_disables_rust(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LITELLM_RUST", value)
    monkeypatch.setenv("LITELLM_USE_RUST_OCR", "1")

    assert configuration.rust_enabled() is False
    assert configuration.rust_ocr_enabled() is False


@pytest.mark.parametrize("value", ("", " ", "sometimes", "2"))
def test_invalid_legacy_environment_value_disables_rust(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LITELLM_USE_RUST_OCR", value)

    with pytest.warns(DeprecationWarning, match="LITELLM_USE_RUST_OCR is deprecated"):
        assert configuration.rust_enabled() is False


def test_process_override_and_reset_apply_to_existing_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "1")

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(configuration.rust_enabled).result() is True
        configuration.rust(False)
        assert executor.submit(configuration.rust_enabled).result() is False
        assert executor.submit(configuration.rust_ocr_enabled).result() is False
        configuration.reset_rust_configuration()
        assert executor.submit(configuration.rust_enabled).result() is True
        assert executor.submit(configuration.rust_ocr_enabled).result() is True


def test_explicit_override_precedes_invalid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "sometimes")

    assert configuration.rust_enabled(request_override=False) is False
    configuration.rust(True)
    assert configuration.rust_enabled() is True


def test_legacy_ocr_environment_is_deprecated_and_global(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_USE_RUST_OCR", "1")

    with pytest.warns(DeprecationWarning, match="LITELLM_USE_RUST_OCR is deprecated"):
        assert configuration.rust_enabled() is True
    with pytest.warns(DeprecationWarning, match="LITELLM_USE_RUST_OCR is deprecated"):
        assert configuration.rust_ocr_enabled() is True


def test_global_environment_precedes_legacy_ocr_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    monkeypatch.setenv("LITELLM_USE_RUST_OCR", "1")

    assert configuration.rust_enabled() is False


@pytest.mark.parametrize("environment_name", ("LITELLM_RUST", "LITELLM_USE_RUST_OCR"))
@pytest.mark.parametrize(("value", "expected"), (("1", "True"), ("0", "False")))
def test_environment_controls_startup(environment_name: str, value: str, expected: str) -> None:
    environment: Final = {**os.environ, environment_name: value}
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
