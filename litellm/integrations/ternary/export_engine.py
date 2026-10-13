"""Ternary FOCUS export engine (polars-backed): one UTC day per upload, with token counts merged into Tags."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from functools import partial
from typing import Final, Literal, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.integrations.focus.database import FocusLiteLLMDatabase
from litellm.integrations.focus.destinations.base import FocusTimeWindow
from litellm.integrations.focus.destinations.ternary_destination import FocusTernaryDestination
from litellm.integrations.focus.export_engine import FocusExportEngine
from litellm.integrations.focus.transformer import FocusTransformer

_TOKEN_TAG_KEYS: Final = (
    "prompt_tokens",
    "completion_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)
_SESSION_TIMEZONE_MARGIN: Final = timedelta(days=1)
_TAGS: Final = TypeAdapter(Mapping[str, object])


class _Series(Protocol):
    def to_list(self) -> Sequence[object]: ...

    def alias(self, name: str) -> _Series: ...


class _Frame(Protocol):
    """The polars DataFrame surface this module uses."""

    @property
    def columns(self) -> Sequence[str]: ...

    @property
    def height(self) -> int: ...

    def is_empty(self) -> bool: ...

    def get_column(self, name: str) -> _Series: ...

    def select(self, columns: Sequence[str], /) -> _Frame: ...

    def hstack(self, columns: _Frame, /) -> _Frame: ...

    def filter(self, mask: Sequence[bool], /) -> _Frame: ...

    def map_rows(self, function: Callable[[tuple[object, ...]], str], /) -> _Frame: ...

    def to_series(self) -> _Series: ...

    def with_columns(self, column: _Series, /) -> _Frame: ...

    def partition_by(
        self, by: str, /, *, maintain_order: bool, as_dict: Literal[True]
    ) -> Mapping[tuple[object, ...], _Frame]: ...


class _UsageSource(Protocol):
    async def get_usage_data(self, *, limit: int | None, start_time_utc: datetime | None = None) -> _Frame: ...


class _Transformer(Protocol):
    def transform(self, frame: _Frame) -> _Frame: ...


class _Serializer(Protocol):
    def serialize(self, frame: _Frame) -> bytes: ...


def _existing_tags(raw: object) -> Mapping[str, object]:
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        return _TAGS.validate_json(raw)
    except ValidationError:
        return {}


def _tags_with_tokens(token_cols: Sequence[str], row: tuple[object, ...]) -> str:
    """Row is (Tags, *token values in `token_cols` order); a missing or non-object Tags starts from empty."""
    tokens: Final = {key: str(value) for key, value in zip(token_cols, row[1:]) if value is not None}
    return json.dumps({**_existing_tags(row[0]), **tokens})


def _merge_token_tags(normalized: _Frame, source: _Frame) -> _Frame:
    """Zip transformed rows with the row-aligned source and add token keys to each row's Tags; degrades to unchanged on any mismatch."""
    if "Tags" not in normalized.columns:
        verbose_logger.warning(
            "Ternary export: transformed data has no `Tags` column; token counts not merged into %d row(s)",
            normalized.height,
        )
        return normalized
    if normalized.height != source.height:
        verbose_logger.warning(
            "Ternary export: row-count mismatch (normalized=%d, source=%d); token counts not merged",
            normalized.height,
            source.height,
        )
        return normalized
    token_cols: Final = tuple(c for c in _TOKEN_TAG_KEYS if c in source.columns)
    if normalized.is_empty() or not token_cols:
        return normalized
    merged: Final = (
        normalized.select(("Tags",)).hstack(source.select(token_cols)).map_rows(partial(_tags_with_tokens, token_cols))
    )
    return normalized.with_columns(merged.to_series().alias("Tags"))


def _canonical_day(raw: object) -> str | None:
    """Return the zero-padded YYYY-MM-DD for a `date` value, or None when it is missing or unparseable."""
    if isinstance(raw, date):
        return raw.isoformat()
    if not isinstance(raw, str):
        return None
    try:
        parsed: Final = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return parsed.date().isoformat()


def _keep_window_days(data: _Frame, window: FocusTimeWindow) -> _Frame:
    """Keep source rows whose UTC `date` falls on a day the window touches; each kept day is then uploaded whole."""
    if "date" not in data.columns:
        return data
    first_day: Final = window.start_time.astimezone(timezone.utc).date().isoformat()
    last_day: Final = (window.end_time.astimezone(timezone.utc) - timedelta(microseconds=1)).date().isoformat()
    kept: Final = data.filter(
        [
            day is not None and first_day <= day <= last_day
            for day in map(_canonical_day, data.get_column("date").to_list())
        ]
    )
    if data.height > 0 and kept.height == 0:
        verbose_logger.warning(
            "Ternary export: day window %s..%s dropped all %d rows (unparseable or out-of-window `date`?)",
            first_day,
            last_day,
            data.height,
        )
    return kept


def _partition_by_day(data: _Frame) -> Sequence[tuple[str, _Frame]]:
    """Split source rows by their UTC `date` (one upload per day); rows with no parseable date are dropped."""
    if "date" not in data.columns:
        verbose_logger.warning("Ternary export: source data has no `date` column; %d row(s) not exported", data.height)
        return ()
    parts: Final = tuple(
        (_canonical_day(key[0]), part)
        for key, part in data.partition_by("date", maintain_order=True, as_dict=True).items()
    )
    undated: Final = sum(part.height for day, part in parts if day is None)
    if undated:
        verbose_logger.warning("Ternary export: dropped %d row(s) with a missing or unparseable `date`", undated)
    return tuple(sorted(((day, part) for day, part in parts if day is not None), key=lambda entry: entry[0]))


def _day_window(day: str, frequency: str) -> FocusTimeWindow:
    """The single UTC day a per-day upload covers."""
    start: Final = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return FocusTimeWindow(start_time=start, end_time=start + timedelta(days=1), frequency=frequency)


async def _read_usage(database: _UsageSource, *, limit: int | None, window: FocusTimeWindow | None) -> _Frame:
    """Read every row, or the rows dated on the days `window` touches.

    `updated_at` is naive UTC filtered as `timestamptz` in the database session timezone, so the read starts one
    day before the window with no upper bound, and each row's own UTC `date` decides what is kept.
    """
    if window is None:
        return await database.get_usage_data(limit=limit)
    data: Final = await database.get_usage_data(
        limit=limit, start_time_utc=window.start_time - _SESSION_TIMEZONE_MARGIN
    )
    return _keep_window_days(data, window)


def _day_payload(data: _Frame, *, transformer: _Transformer, serializer: _Serializer) -> bytes:
    transformed: Final = transformer.transform(data)
    if transformed.is_empty():
        return b""
    return serializer.serialize(_merge_token_tags(transformed, data))


class TernaryExportEngine(FocusExportEngine):
    """FocusExportEngine that uploads one whole UTC day per delivery, with token counts merged into Tags."""

    def __init__(self, *, prefix: str, destination: FocusTernaryDestination) -> None:
        """Set the base engine's fields directly, since the generic destination factory has no Ternary destination."""
        self.provider = "ternary"
        self.export_format = "csv"
        self.prefix = prefix
        self.destination = destination
        self.serializer = self._init_serializer()
        self.transformer = FocusTransformer()
        self.database = FocusLiteLLMDatabase()

    async def export_window(self, *, window: FocusTimeWindow, limit: int | None) -> None:
        data: Final = await _read_usage(self._database, limit=limit, window=window)
        await self._deliver_enriched(data, window)

    async def export_all(self, *, limit: int | None) -> None:
        data: Final = await _read_usage(self._database, limit=limit, window=None)
        now: Final = datetime.now(timezone.utc)
        window: Final = FocusTimeWindow(
            start_time=now.replace(hour=0, minute=0, second=0, microsecond=0),
            end_time=now,
            frequency="all",
        )
        await self._deliver_enriched(data, window)

    async def _deliver_enriched(self, data: _Frame, window: FocusTimeWindow) -> None:
        if data.is_empty():
            verbose_logger.debug("Ternary export: no usage data for window %s", window)
            return
        failed: Final[list[str]] = []  # mutable-ok: days whose upload failed, reported once after all days run
        for day, day_data in _partition_by_day(data):
            try:
                await self._deliver_day(day_data, _day_window(day, window.frequency))
            except Exception as e:  # noqa: BLE001  # one day's failure must not block the rest; logged and re-raised
                verbose_logger.warning("Ternary export: upload for %s failed: %s", day, e)
                failed.append(day)
        if failed:
            raise RuntimeError(f"Ternary export: {len(failed)} day(s) failed to upload: {', '.join(failed)}")

    async def _deliver_day(self, data: _Frame, window: FocusTimeWindow) -> None:
        payload: Final = _day_payload(data, transformer=self._transformer, serializer=self._serializer)
        if not payload:
            verbose_logger.debug("Ternary export: nothing to upload for %s", window.start_time.date())
            return
        await self._destination.deliver(
            content=payload,
            time_window=window,
            filename=self._build_filename(window),
        )


__all__ = ("TernaryExportEngine",)
