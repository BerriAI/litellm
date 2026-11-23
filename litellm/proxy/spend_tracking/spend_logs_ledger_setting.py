from collections.abc import Callable, Mapping
from typing import Final


class _StartupSetting:
    __slots__ = ("captured",)

    def __init__(self) -> None:
        self.captured: bool | None = None


_STARTUP: Final = _StartupSetting()


def _enabled_in(settings: Mapping[str, object]) -> bool:
    return settings.get("use_spend_logs_ledger", True) is not False


def capture_spend_logs_ledger_setting(settings: Mapping[str, object]) -> Callable[[], None]:
    _STARTUP.captured = _enabled_in(settings)

    def forget() -> None:
        _STARTUP.captured = None

    return forget


def spend_logs_ledger_enabled(live_settings: Mapping[str, object]) -> bool:
    captured: Final = _STARTUP.captured
    return _enabled_in(live_settings) if captured is None else captured


__all__ = ["capture_spend_logs_ledger_setting", "spend_logs_ledger_enabled"]
