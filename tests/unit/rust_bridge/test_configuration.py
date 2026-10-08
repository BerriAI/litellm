from __future__ import annotations

from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor

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


@pytest.mark.parametrize("process", (None, False, True))
def test_process_override_applies_when_environment_is_unset(process: bool | None) -> None:
    configuration.rust(process)

    assert configuration.rust_enabled() is (process is True)


@pytest.mark.parametrize("value", ("", " ", "sometimes", "2"))
def test_invalid_environment_value_is_ignored(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LITELLM_RUST", value)

    assert configuration.rust_enabled() is False
    configuration.rust(True)
    assert configuration.rust_enabled() is True


def test_process_override_and_reset_apply_to_existing_threads() -> None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(configuration.rust_enabled).result() is False
        configuration.rust(True)
        assert executor.submit(configuration.rust_enabled).result() is True
        configuration.reset_rust_configuration()
        assert executor.submit(configuration.rust_enabled).result() is False
