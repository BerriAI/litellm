"""Ternary logger — thin wrapper around the Focus export pipeline.

Configures FocusLogger to use the Ternary API destination with CSV format so
users can simply set ``callbacks: ["ternary"]`` in their proxy config.

Beyond presetting the destination, this
logger enriches the FOCUS ``Tags`` column with per-row token counts that the
shared FocusTransformer drops. Ternary weights cost allocation by token
consumption, so the token breakdown must survive the export. The enrichment is
confined to this Ternary-only code path and only *adds* keys to ``Tags``
(FOCUS v1.2's escape hatch for non-standard fields) — it never modifies the
shared transformer, so other FOCUS exports are unaffected.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final, TypeAlias

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.focus.destinations.base import FocusTimeWindow
from litellm.integrations.focus.focus_logger import FocusLogger

if TYPE_CHECKING:
    import polars as pl
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    from litellm.integrations.focus.export_engine import FocusExportEngine
else:
    AsyncIOScheduler: TypeAlias = object

TERNARY_USAGE_DATA_JOB_NAME: Final = "ternary_export_usage_data"

# No "hourly": spend is a daily aggregate under whole-day replace, so sub-daily adds no grain.
_SUPPORTED_FREQUENCIES: Final = frozenset({"daily", "interval"})

# Raw LiteLLM daily-spend token columns carried in the FOCUS Tags JSON.
_TOKEN_TAG_KEYS: Final = (
    "prompt_tokens",
    "completion_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _merge_token_tags(normalized: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    """Merge raw token counts from the source rows into the FOCUS Tags JSON.

    The transformer emits rows 1:1 and row-aligned with the source, so we zip
    by position and add the token keys to each row's existing Tags dict.
    Degrades gracefully: on a row-count mismatch, missing token columns, or a
    malformed Tags value, the data is left unchanged rather than failing the
    export.
    """
    import polars as pl  # local import: polars is a heavy optional dependency

    if "Tags" not in normalized.columns or normalized.height != source.height:
        return normalized
    token_cols: Final = tuple(c for c in _TOKEN_TAG_KEYS if c in source.columns)
    if not token_cols:
        return normalized

    token_rows: Final = source.select(token_cols).to_dicts()
    tags_values: Final = normalized["Tags"].to_list()

    merged: Final[list[str]] = []  # mutable-ok: per-row accumulator built in the zip loop
    for tags_json, tokens in zip(tags_values, token_rows):
        try:
            parsed = json.loads(tags_json) if tags_json else {}  # mutable-ok: fresh per-row tag dict
            if not isinstance(parsed, dict):
                parsed = {}  # mutable-ok: non-object Tags degrades to empty
        except (json.JSONDecodeError, TypeError):
            parsed = {}  # mutable-ok: malformed Tags degrades to empty
        for key, value in tokens.items():
            if value is not None:
                parsed[key] = str(value)
        merged.append(json.dumps(parsed))

    return normalized.with_columns(pl.Series("Tags", merged))


def _drop_days_before(data: pl.DataFrame, floor: datetime) -> pl.DataFrame:
    """Drop source rows for days older than the window start.

    The receiver replaces cost by whole UTC day (``ChargePeriodStart``), but
    ``get_usage_data`` filters by ``updated_at`` -- so a row for an *older* day that
    was merely touched inside the window would partially replace that day and
    truncate it. Keeping only days at or after the window start means every day we
    send is sent in full, so replace-by-day never truncates. (A backlog draining for
    older days is consequently not delivered by the scheduled path; that needs a
    wider/backfill window, by design.)
    """
    import polars as pl  # local import: polars is a heavy optional dependency

    if "date" not in data.columns:
        return data
    floor_date: Final = floor.astimezone(timezone.utc).date()
    date_col: Final = pl.col("date").cast(pl.Utf8)  # cast-ok: polars dtype cast, not typing.cast
    parsed: Final = date_col.str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    kept: Final = data.filter(parsed >= floor_date)
    if data.height > 0 and kept.height == 0:
        verbose_logger.warning(
            "Ternary export: day-window floor %s dropped all %d rows (unparseable or older `date`?)",
            floor_date,
            data.height,
        )
    return kept


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
    """FocusLogger pre-configured for Ternary (CSV format, Ternary cost-import API).

    Environment Variables:
        TERNARY_API_KEY: per-connection shared secret for the Ternary receiver
        TERNARY_CONNECTION_ID: external-cost-source connection id (used in the URL path)
        TERNARY_BASE_URL: required — the Ternary API host (region-specific; no default)
        TERNARY_EXPORT_FREQUENCY: export cadence — "daily" (default) or "interval"
            (short setup-validation / test loops only). "hourly" is unsupported: LiteLLM
            spend is a daily aggregate landed with whole-day replace, so it adds no grain.
        TERNARY_EXPORT_INTERVAL_SECONDS: interval in seconds when frequency is "interval"
    """

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

        destination_config: Final[dict[str, str]] = {}  # mutable-ok: built from the config values present below
        if resolved_api_key:
            destination_config["api_key"] = resolved_api_key
        if resolved_connection_id:
            destination_config["connection_id"] = resolved_connection_id
        if resolved_base_url:
            destination_config["base_url"] = resolved_base_url

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
        """Snap the window start to the previous UTC midnight so each push carries
        complete days -- the receiver replaces landed cost by whole day."""
        now_utc: Final = now.astimezone(timezone.utc)
        start_time: Final = (now_utc - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return FocusTimeWindow(start_time=start_time, end_time=now_utc, frequency=self.frequency)

    async def _export_window(self, *, window: FocusTimeWindow, limit: int | None) -> None:
        engine: Final = self._ensure_engine()
        data: Final = await engine._database.get_usage_data(
            limit=limit,
            start_time_utc=window.start_time,
            end_time_utc=window.end_time,
        )
        windowed: Final = _drop_days_before(data, window.start_time)
        await self._transform_enrich_deliver(engine=engine, data=windowed, window=window)

    async def _export_all(self, *, limit: int | None) -> None:
        engine: Final = self._ensure_engine()
        data: Final = await engine._database.get_usage_data(limit=limit)
        now: Final = datetime.now(timezone.utc)
        window: Final = FocusTimeWindow(
            start_time=now.replace(hour=0, minute=0, second=0, microsecond=0),
            end_time=now,
            frequency="all",
        )
        await self._transform_enrich_deliver(engine=engine, data=data, window=window)

    async def _transform_enrich_deliver(
        self,
        *,
        engine: FocusExportEngine,
        data: pl.DataFrame,
        window: FocusTimeWindow,
    ) -> None:
        if data.is_empty():
            verbose_logger.debug("Ternary export: no usage data for window %s", window)
            return
        transformed: Final = engine._transformer.transform(data)
        if transformed.is_empty():
            verbose_logger.debug("Ternary export: normalized data empty for window %s", window)
            return
        enriched: Final = _merge_token_tags(transformed, data)
        payload: Final = engine._serializer.serialize(enriched)
        if not payload:
            verbose_logger.debug("Ternary export: serializer returned empty payload")
            return
        await engine._destination.deliver(
            content=payload,
            time_window=window,
            filename=engine._build_filename(window),
        )

    async def initialize_focus_export_job(self) -> None:
        """Override to use a Ternary-specific pod lock key.

        Without this, TernaryLogger and FocusLogger would compete for the same
        ``FOCUS_USAGE_DATA_JOB_NAME`` lock, causing one to silently skip its
        export cycle when both are configured simultaneously.
        """
        from litellm.proxy.proxy_server import proxy_logging_obj

        writer: Final = getattr(proxy_logging_obj, "db_spend_update_writer", None) if proxy_logging_obj else None
        pod_lock_manager: Final = getattr(writer, "pod_lock_manager", None) if writer is not None else None

        if pod_lock_manager and pod_lock_manager.redis_cache:
            acquired: Final = await pod_lock_manager.acquire_lock(cronjob_id=TERNARY_USAGE_DATA_JOB_NAME)
            if not acquired:
                verbose_logger.debug("Ternary export: unable to acquire pod lock")
                return
            try:
                await self._run_scheduled_export()
            finally:
                await pod_lock_manager.release_lock(cronjob_id=TERNARY_USAGE_DATA_JOB_NAME)
        else:
            await self._run_scheduled_export()

    @staticmethod
    async def init_ternary_background_job(
        scheduler: AsyncIOScheduler,
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
