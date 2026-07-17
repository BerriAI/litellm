"""MavvrikFocusLogger — FOCUS-based Mavvrik export logger.

Usage in config.yaml:
    litellm_settings:
      callbacks: ["mavvrik"]

Required env vars:
    MAVVRIK_API_KEY
    MAVVRIK_API_ENDPOINT
    MAVVRIK_CONNECTION_ID

Optional env vars:
    MAVVRIK_FOCUS_MAX_ROWS  — row cap per export window (default: 500000)

Only daily frequency is supported. The Mavvrik ingestion protocol stores one
file per calendar date (metrics/YYYY-MM-DD). Hourly or interval exports would
overwrite each other within the same day, producing incomplete data.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, List, Optional

import polars as pl

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import MAVVRIK_FOCUS_EXPORT_JOB_NAME
from litellm.integrations.focus.destinations.base import FocusTimeWindow
from litellm.integrations.focus.focus_logger import FocusLogger

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
else:
    AsyncIOScheduler = Any

# FOCUS v1.2 has no standard column for token counts; core's transformer
# drops prompt_tokens/completion_tokens even though the source query selects
# them. Mavvrik carries them through as extra keys in the existing Tags JSON
# column (the spec's own escape hatch for non-standard fields), rather than
# changing the shared transformer used by every FOCUS destination. total_tokens
# isn't a stored column at all -- it's derived here as their sum.
_TOKEN_TAG_KEYS = ("prompt_tokens", "completion_tokens")


def _with_token_tags(data: pl.DataFrame, normalized: pl.DataFrame) -> pl.DataFrame:
    """Merge prompt/completion token counts (and their sum, total_tokens) from
    the pre-transform frame into ``normalized``'s Tags column. Rows correspond
    1:1 and in the same order across both frames -- transform() only
    adds/renames columns, it never filters or reorders rows.
    """
    available = [k for k in _TOKEN_TAG_KEYS if k in data.columns]
    if not available or len(data) != len(normalized):
        return normalized

    token_rows = data.select(available).to_dicts()
    has_both = "prompt_tokens" in available and "completion_tokens" in available

    def _merge(tags_json: str, row: dict) -> str:
        try:
            tags = json.loads(tags_json) if tags_json else {}
        except (TypeError, ValueError):
            tags = {}
        for key in available:
            value = row.get(key)
            if value is not None:
                tags[key] = str(value)
        if has_both:
            prompt = row.get("prompt_tokens")
            completion = row.get("completion_tokens")
            if prompt is not None and completion is not None:
                tags["total_tokens"] = str(prompt + completion)
        return json.dumps(tags)

    merged_tags = pl.Series(
        [_merge(tags_json, row) for tags_json, row in zip(normalized["Tags"].to_list(), token_rows)]
    )
    return normalized.with_columns(merged_tags.alias("Tags"))


def _parse_metrics_marker(
    marker: Optional[object],
) -> Optional[datetime]:
    """Parse metricsMarker from Mavvrik register response into a UTC datetime.

    Handles both formats Mavvrik may return:
    - Unix timestamp (int/float): e.g. 1749340800
    - ISO date string: e.g. "2026-06-09" or "2026-06-09T00:00:00Z"

    Returns None for falsy values (0, None, empty string) which indicate
    no data has been ingested yet.
    """
    if not marker:
        return None
    try:
        if isinstance(marker, (int, float)):
            return datetime.fromtimestamp(float(marker), tz=timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        if isinstance(marker, str):
            marker = marker.strip()
            if not marker:
                return None
            # Try ISO date first (YYYY-MM-DD), then full ISO datetime
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
                try:
                    return datetime.strptime(marker, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
    except Exception:
        pass
    verbose_proxy_logger.warning("Mavvrik FOCUS: could not parse metricsMarker %r — skipping catch-up", marker)
    return None


def _is_empty_metrics_marker(marker: Optional[object]) -> bool:
    if marker is None:
        return True
    if isinstance(marker, (int, float)):
        return marker == 0
    if isinstance(marker, str):
        return not marker.strip()
    return False


class MavvrikFocusLogger(FocusLogger):
    """FOCUS-based export logger that routes to the Mavvrik destination."""

    def __init__(self, **kwargs: Any) -> None:
        frequency = os.getenv("MAVVRIK_FOCUS_FREQUENCY", "daily").lower()
        if frequency != "daily":
            raise ValueError(
                f"MAVVRIK_FOCUS_FREQUENCY='{frequency}' is not supported. "
                "Only 'daily' is allowed -- the Mavvrik ingestion protocol stores one "
                "file per calendar date (metrics/YYYY-MM-DD). Hourly or interval "
                "exports would overwrite each other within the same day."
            )
        super().__init__(
            provider="mavvrik",
            export_format="csv",
            frequency="daily",
            prefix="mavvrik_focus_exports",
            destination_config={
                "api_key": os.getenv("MAVVRIK_API_KEY"),
                "api_endpoint": os.getenv("MAVVRIK_API_ENDPOINT"),
                "connection_id": os.getenv("MAVVRIK_CONNECTION_ID"),
            },
            **kwargs,
        )
        raw = os.getenv("MAVVRIK_FOCUS_MAX_ROWS")
        self._max_rows: Optional[int] = int(raw) if raw else 500_000

    async def _export_window(
        self,
        *,
        window: FocusTimeWindow,
        limit: Optional[int],
    ) -> None:
        """Export with Mavvrik row cap applied when no explicit limit is passed."""
        effective_limit = limit if limit is not None else self._max_rows
        engine = self._ensure_engine()
        data = await engine._database.get_usage_data(
            limit=effective_limit,
            start_time_utc=window.start_time,
            end_time_utc=window.end_time,
        )
        if effective_limit is not None and len(data) >= effective_limit:
            verbose_proxy_logger.warning(
                "Mavvrik FOCUS export: row cap reached (%d rows). "
                "Some data for window %s→%s may be excluded. "
                "Increase MAVVRIK_FOCUS_MAX_ROWS to export all rows.",
                effective_limit,
                window.start_time.date(),
                window.end_time.date(),
            )
        payload = b""
        if data.is_empty():
            verbose_proxy_logger.debug("Mavvrik FOCUS export: no usage data for window %s", window)
        else:
            normalized = engine._transformer.transform(data)
            if not normalized.is_empty():
                normalized = _with_token_tags(data, normalized)
                payload = engine._serializer.serialize(normalized)
        await engine._destination.deliver(
            content=payload or b"",
            time_window=window,
            filename=engine._build_filename(window),
        )

    # Maximum number of days to catch up in a single run. Prevents runaway
    # loops if the connector was disabled for a long time, and avoids querying
    # data that has likely been cleaned up from LiteLLM_DailyUserSpend.
    _MAX_CATCHUP_DAYS = 7

    async def _run_scheduled_export(self) -> None:
        """Export today's window, catching up any dates Mavvrik has not yet received.

        On each run:
        1. Register with Mavvrik → get metricsMarker (last successfully ingested date)
        2. If metricsMarker is behind yesterday (or 0/None for a fresh connector),
           catch up missed dates (capped at _MAX_CATCHUP_DAYS)
        3. Export yesterday (today's daily window)

        This ensures a failed export on day N is automatically retried on day N+1
        without any manual intervention.
        """
        engine = self._ensure_engine()
        from litellm.integrations.focus.destinations.mavvrik_destination import (  # noqa: PLC0415
            FocusMavvrikDestination,
        )

        destination = engine._destination
        if not isinstance(destination, FocusMavvrikDestination):
            await super()._run_scheduled_export()
            return

        # Register and get the last date Mavvrik has processed.
        # metricsMarker may be a Unix timestamp (int/float) or an ISO date string.
        marker = await destination.get_metrics_marker()

        now = datetime.now(timezone.utc)
        yesterday = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)

        last_ingested = _parse_metrics_marker(marker)

        is_empty_marker = _is_empty_metrics_marker(marker)
        earliest_catchup = yesterday - timedelta(days=self._MAX_CATCHUP_DAYS - 1)
        if is_empty_marker or (last_ingested is not None and last_ingested < yesterday):
            catch_up_date = (
                earliest_catchup if last_ingested is None else max(last_ingested + timedelta(days=1), earliest_catchup)
            )

            if last_ingested is not None and last_ingested + timedelta(days=1) < earliest_catchup:
                verbose_proxy_logger.warning(
                    "Mavvrik FOCUS export: metricsMarker is more than %d days behind "
                    "(%s). Catching up from %s only; earlier data will not be re-exported.",
                    self._MAX_CATCHUP_DAYS,
                    last_ingested.date(),
                    catch_up_date.date(),
                )

            while catch_up_date < yesterday:
                verbose_proxy_logger.info(
                    "Mavvrik FOCUS export: catching up missed date %s",
                    catch_up_date.date(),
                )
                # Use now as end_time for catch-up windows too — rows for old dates
                # may have been flushed to DB well after their calendar day ended.
                catch_up_end = min(catch_up_date + timedelta(days=1), now)
                window = FocusTimeWindow(
                    start_time=catch_up_date,
                    end_time=catch_up_end,
                    frequency="daily",
                )
                await self._export_window(window=window, limit=None)
                catch_up_date += timedelta(days=1)

        # Export yesterday's window (the normal daily run).
        # Use `now` as end_time so spend rows flushed after midnight are included.
        # LiteLLM's DailyUserSpend rows for a given date keep getting updated_at
        # bumped as the flush job runs; capping at midnight would miss those updates.
        window = FocusTimeWindow(
            start_time=yesterday,
            end_time=now,
            frequency="daily",
        )
        await self._export_window(window=window, limit=None)

    async def initialize_mavvrik_focus_export_job(self) -> None:
        """Scheduler entry point — uses Mavvrik-specific pod-lock key."""
        from litellm.proxy.proxy_server import proxy_logging_obj  # noqa: PLC0415

        pod_lock_manager = None
        if proxy_logging_obj is not None:
            writer = getattr(proxy_logging_obj, "db_spend_update_writer", None)
            if writer is not None:
                pod_lock_manager = getattr(writer, "pod_lock_manager", None)

        if pod_lock_manager and pod_lock_manager.redis_cache:
            acquired = await pod_lock_manager.acquire_lock(cronjob_id=MAVVRIK_FOCUS_EXPORT_JOB_NAME)
            if not acquired:
                verbose_proxy_logger.debug("Mavvrik FOCUS export: unable to acquire pod lock")
                return
            try:
                await self._run_scheduled_export()
            finally:
                await pod_lock_manager.release_lock(cronjob_id=MAVVRIK_FOCUS_EXPORT_JOB_NAME)
        else:
            await self._run_scheduled_export()

    @staticmethod
    async def init_mavvrik_focus_background_job(
        scheduler: AsyncIOScheduler,
    ) -> None:
        """Register the Mavvrik FOCUS export job on the provided scheduler."""
        loggers: List[MavvrikFocusLogger] = [
            cb
            for cb in litellm.logging_callback_manager.get_custom_loggers_for_type(callback_type=MavvrikFocusLogger)
            if type(cb) is MavvrikFocusLogger
        ]
        if not loggers and "mavvrik" in litellm.callbacks:
            # The logger is registered as the string "mavvrik" but hasn't been
            # instantiated yet (lazy init happens on first LLM call). Force it now
            # so the scheduler can register the daily export job at startup.
            from litellm.litellm_core_utils.litellm_logging import (  # noqa: PLC0415
                _init_custom_logger_compatible_class,
            )

            instance = _init_custom_logger_compatible_class(
                logging_integration="mavvrik",
                internal_usage_cache=None,
                llm_router=None,
            )
            if isinstance(instance, MavvrikFocusLogger):
                loggers = [instance]
        if not loggers:
            verbose_proxy_logger.debug("No MavvrikFocusLogger registered; skipping scheduler")
            return

        logger = loggers[0]
        trigger_kwargs = logger._build_scheduler_trigger()
        scheduler.add_job(  # type: ignore[attr-defined]
            logger.initialize_mavvrik_focus_export_job,
            id=MAVVRIK_FOCUS_EXPORT_JOB_NAME,
            replace_existing=True,
            **trigger_kwargs,
        )
        verbose_proxy_logger.info("mavvrik_focus: background export job scheduled (%s)", trigger_kwargs)
