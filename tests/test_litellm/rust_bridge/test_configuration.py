from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Generator
from typing import Final

import pytest

from litellm.rust_bridge import configuration
from litellm.rust_bridge import ocr as rust_ocr


class _OcrBridge:
    def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]:
        return {}


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
    ("request_override", "process", "environment", "legacy_ocr", "release_default", "expected"),
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
    legacy_ocr: bool | None,
    release_default: bool,
    expected: bool,
) -> None:
    assert (
        configuration.resolve_rust_enabled(
            request_override=request_override,
            process_override=process,
            environment_override=environment,
            legacy_ocr_override=legacy_ocr,
            release_default=release_default,
        )
        is expected
    )


def test_release_default_remains_disabled() -> None:
    assert configuration.DEFAULT_RUST_ENABLED is False
    assert configuration.rust_enabled() is False


def test_process_override_wins_over_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    configuration.use_litellm_rust(True)

    assert configuration.rust_enabled() is True
    assert configuration.rust_enabled(request_override=False) is False


def test_global_environment_accepts_explicit_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "off")

    assert configuration.rust_enabled() is False


def test_invalid_environment_value_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "sometimes")

    with pytest.raises(ValueError, match="LITELLM_RUST must be one of"):
        configuration.rust_enabled()


def test_explicit_override_precedes_invalid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "sometimes")

    assert configuration.rust_enabled(request_override=False) is False
    configuration.use_litellm_rust(True)
    assert configuration.rust_enabled() is True


def test_legacy_ocr_environment_is_deprecated_and_ocr_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_USE_RUST_OCR", "1")

    with pytest.warns(DeprecationWarning, match="LITELLM_USE_RUST_OCR is deprecated"):
        assert configuration.rust_ocr_enabled() is True
    assert configuration.rust_enabled() is False


def test_global_environment_precedes_legacy_ocr_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    monkeypatch.setenv("LITELLM_USE_RUST_OCR", "1")

    assert configuration.rust_ocr_enabled() is False


def test_deprecated_public_injection_delegates_to_internal_binding() -> None:
    bridge: Final = _OcrBridge()

    with pytest.warns(DeprecationWarning, match="Injecting Rust bridge implementations"):
        configuration.use_litellm_rust(True, ocr=bridge)

    assert rust_ocr.load_rust_ocr() is bridge


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
