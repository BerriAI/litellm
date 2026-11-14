"""Ternary FocusLogger preset (CSV, Ternary destination) that adds per-row token counts to the FOCUS Tags."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, ClassVar, Final, Protocol

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.focus.destinations.base import FocusTimeWindow
from litellm.integrations.focus.destinations.ternary_destination import FocusTernaryDestination
from litellm.integrations.focus.focus_logger import FocusLogger

if TYPE_CHECKING:
    from litellm.integrations.focus.export_engine import FocusExportEngine

TERNARY_USAGE_DATA_JOB_NAME: Final = "ternary_export_usage_data"

_SUPPORTED_FREQUENCIES: Final = frozenset({"daily", "interval"})


class _JobScheduler(Protocol):
    """The slice of ``AsyncIOScheduler`` this module uses, which ships no type information"""

    def add_job(
        self,
        func: Callable[[], Awaitable[None]],
        /,
        **trigger: str | int,  # kwargs-ok: APScheduler add_job trigger options
    ) -> object: ...


def _parse_interval(raw: str | int | None) -> int | None:
    """Parse the export interval-seconds override; a non-numeric value is ignored."""
    if raw is None:
        return None
    try:
        return int(raw)
    except (ValueError, TypeError):
        verbose_logger.warning("Invalid TERNARY_EXPORT_INTERVAL_SECONDS value: %s, ignoring", raw)
        return None


class TernaryLogger(FocusLogger):
    """FocusLogger preset for Ternary (CSV, Ternary cost-import API).

    Env: TERNARY_API_KEY, TERNARY_CONNECTION_ID, TERNARY_BASE_URL (required),
    TERNARY_EXPORT_FREQUENCY ("daily" default / "interval"), TERNARY_EXPORT_INTERVAL_SECONDS.
    """

    export_job_name: ClassVar[str] = TERNARY_USAGE_DATA_JOB_NAME

    def __init__(
        self,
        *,
        api_key: str | None = None,
        connection_id: str | None = None,
        base_url: str | None = None,
        frequency: str | None = None,
        interval_seconds: int | None = None,
    ) -> None:
        resolved_api_key: Final = api_key or os.getenv("TERNARY_API_KEY")
        resolved_connection_id: Final = connection_id or os.getenv("TERNARY_CONNECTION_ID")
        resolved_base_url: Final = base_url or os.getenv("TERNARY_BASE_URL")
        resolved_frequency: Final = (frequency or os.getenv("TERNARY_EXPORT_FREQUENCY") or "daily").lower()
        if resolved_frequency not in _SUPPORTED_FREQUENCIES:
            raise ValueError(
                f"Unsupported TERNARY_EXPORT_FREQUENCY {resolved_frequency!r}; "
                f"Ternary supports {sorted(_SUPPORTED_FREQUENCIES)}. LiteLLM spend is a daily "
                "aggregate landed with whole-day replace, so 'hourly' adds no grain -- use "
                "'interval' only for short setup-validation loops."
            )

        resolved_interval: Final = _parse_interval(
            interval_seconds if interval_seconds is not None else os.getenv("TERNARY_EXPORT_INTERVAL_SECONDS")
        )
        if resolved_frequency == "interval" and (resolved_interval is None or resolved_interval <= 0):
            raise ValueError(
                "TERNARY_EXPORT_INTERVAL_SECONDS must be a positive integer when TERNARY_EXPORT_FREQUENCY is 'interval'"
            )

        destination_config: Final = {
            key: value
            for key, value in (
                ("api_key", resolved_api_key),
                ("connection_id", resolved_connection_id),
                ("base_url", resolved_base_url),
            )
            if value
        }

        super().__init__(
            provider="ternary",
            export_format="csv",
            frequency=resolved_frequency,
            interval_seconds=resolved_interval,
            prefix="ternary_exports",
            destination_config=destination_config,
        )

        verbose_logger.debug(
            "TernaryLogger initialized (connection_id=%s)",
            (
                resolved_connection_id[:4] + "***"
                if resolved_connection_id and len(resolved_connection_id) > 4
                else "***"
            ),
        )

    def _compute_time_window(self, now: datetime) -> FocusTimeWindow:
        """Start at UTC midnight two days back so each push re-sends whole days, including spend flushed late."""
        now_utc: Final = now.astimezone(timezone.utc)
        start_time: Final = (now_utc - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
        return FocusTimeWindow(start_time=start_time, end_time=now_utc, frequency=self.frequency)

    def _ensure_engine(self) -> FocusExportEngine:
        """Instantiate the polars-backed Ternary export engine lazily."""
        if self._engine is None:
            from .export_engine import TernaryExportEngine  # noqa: PLC0415  # polars is a proxy extra

            self._engine = TernaryExportEngine(
                prefix=self.prefix,
                destination=FocusTernaryDestination(prefix=self.prefix, config=self._destination_config),
            )
        return self._engine

    @staticmethod
    async def init_ternary_background_job(
        scheduler: _JobScheduler,
    ) -> None:
        """Register the Ternary export job with the provided scheduler."""
        ternary_loggers: Final[list[CustomLogger]] = (  # mutable-ok: list returned by the shared callback manager
            litellm.logging_callback_manager.get_custom_loggers_for_type(callback_type=TernaryLogger)
        )
        if not ternary_loggers:
            verbose_logger.debug("No Ternary logger registered; skipping scheduler")
            return

        ternary_logger: Final = ternary_loggers[0]
        if not isinstance(ternary_logger, TernaryLogger):
            return
        trigger_kwargs: Final = ternary_logger._build_scheduler_trigger()
        scheduler.add_job(
            ternary_logger.initialize_focus_export_job,
            **trigger_kwargs,
        )


__all__ = ("TernaryLogger",)
