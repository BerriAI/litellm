from __future__ import annotations

import warnings
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor

import pytest

from litellm.rust_bridge import configuration


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    configuration.reset_rust_configuration()
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    monkeypatch.delenv("LITELLM_USE_RUST_OCR", raising=False)
    yield
    configuration.reset_rust_configuration()


@pytest.mark.parametrize(
    ("request_override", "process", "environment", "legacy_environment", "release_default", "expected"),
    (
        pytest.param(False, True, True, True, True, False, id="request-false-wins"),
        pytest.param(True, False, False, False, False, True, id="request-true-wins"),
        pytest.param(None, False, True, True, True, False, id="process-false-wins"),
        pytest.param(None, True, False, False, False, True, id="process-true-wins"),
        pytest.param(None, None, False, True, True, False, id="environment-false-wins"),
        pytest.param(None, None, True, False, False, True, id="environment-true-wins"),
        pytest.param(None, None, None, False, True, False, id="legacy-false-wins"),
        pytest.param(None, None, None, True, False, True, id="legacy-true-wins"),
        pytest.param(None, None, None, None, False, False, id="release-default-false"),
        pytest.param(None, None, None, None, True, True, id="release-default-true"),
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


@pytest.mark.parametrize(
    ("environment_name", "value", "expected", "deprecated"),
    tuple(
        pytest.param(environment_name, value, expected, environment_name == "LITELLM_USE_RUST_OCR", id=case_id)
        for environment_name, prefix in (
            ("LITELLM_RUST", "global"),
            ("LITELLM_USE_RUST_OCR", "legacy"),
        )
        for value, expected, case_id in (
            ("1", True, f"{prefix}-one"),
            ("TRUE", True, f"{prefix}-true"),
            (" yes ", True, f"{prefix}-yes-trimmed"),
            ("on", True, f"{prefix}-on"),
            ("0", False, f"{prefix}-zero"),
            ("false", False, f"{prefix}-false"),
            ("", False, f"{prefix}-empty"),
            ("sometimes", False, f"{prefix}-invalid"),
        )
    ),
)
def test_environment_values(
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
    value: str,
    expected: bool,
    deprecated: bool,
) -> None:
    monkeypatch.setenv(environment_name, value)

    if deprecated:
        with pytest.warns(DeprecationWarning, match="LITELLM_USE_RUST_OCR is deprecated"):
            assert configuration.rust_enabled() is expected
        return
    assert configuration.rust_enabled() is expected


def test_release_default_is_disabled() -> None:
    assert configuration.rust_enabled() is False


def test_request_and_process_overrides_precede_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    configuration.rust(True)

    assert configuration.rust_enabled() is True
    assert configuration.rust_enabled(request_override=False) is False


def test_global_environment_precedes_legacy_without_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    monkeypatch.setenv("LITELLM_USE_RUST_OCR", "1")

    with warnings.catch_warnings(record=True) as caught:
        assert configuration.rust_enabled() is False

    assert caught == []


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
