from collections.abc import Iterator
from typing import Final

import pytest

import litellm.proxy.proxy_server as ps
from litellm.proxy.spend_tracking.rust_spend_log_rows import uses_spend_logs_ledger
from litellm.proxy.spend_tracking.spend_logs_ledger_setting import (
    capture_spend_logs_ledger_setting,
    spend_logs_ledger_enabled,
)

_LEDGER_OFF: Final = {"use_spend_logs_ledger": False}
_LEDGER_ON: Final = {"use_spend_logs_ledger": True}


@pytest.fixture
def captured_off() -> Iterator[None]:
    forget: Final = capture_spend_logs_ledger_setting(_LEDGER_OFF)
    try:
        yield
    finally:
        forget()


@pytest.mark.usefixtures("captured_off")
def test_the_setting_captured_at_startup_outlives_later_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ps, "general_settings", dict(_LEDGER_ON))

    assert spend_logs_ledger_enabled(_LEDGER_ON) is False
    assert uses_spend_logs_ledger() is False


@pytest.mark.parametrize(("settings", "enabled"), [({}, True), (_LEDGER_OFF, False), (_LEDGER_ON, True)])
def test_without_a_startup_capture_the_live_setting_decides(
    monkeypatch: pytest.MonkeyPatch, settings: dict[str, bool], enabled: bool
) -> None:
    capture_spend_logs_ledger_setting(_LEDGER_OFF)()
    monkeypatch.setattr(ps, "general_settings", dict(settings))

    assert spend_logs_ledger_enabled(settings) is enabled
    assert uses_spend_logs_ledger() is enabled
