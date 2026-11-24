import csv
import io
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime
from typing import Final

import pytest

from litellm.models.plugin_upload import PluginDownloadEvent
from litellm.proxy.discovery_endpoints.skill_download_export import _stream_downloads

EVENT: Final = PluginDownloadEvent(
    id="9007199254740993",
    downloaded_at=datetime.fromisoformat("2026-10-01T12:00:00+02:00"),
    version="1.0",
    digest="a" * 64,
    authenticated=True,
    user_id="reader",
    user_email="reader@example.test",
    key_alias='clé, "quoted"\nnext',
    team_id="team-id",
    team_alias=None,
)


async def _events(*events: PluginDownloadEvent) -> AsyncIterator[PluginDownloadEvent]:
    for event in events:
        yield event


@pytest.mark.asyncio
async def test_csv_preserves_attribution_unicode_escaping_and_utc_timestamps() -> None:
    content: Final = b"".join([chunk async for chunk in _stream_downloads(EVENT, _events())]).decode()
    assert list(csv.DictReader(io.StringIO(content))) == [
        {
            "download_id": EVENT.id,
            "downloaded_at": "2026-10-01T10:00:00.000Z",
            "user_email": EVENT.user_email,
            "user_id": EVENT.user_id,
            "team": "team-id",
            "key": EVENT.key_alias,
            "version": EVENT.version,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("formula", ("=SUM(1,2)", "+123", "-123", "@name", "\tformula", "\rformula", "=1+\n1"))
async def test_formula_cells_are_neutralized_in_every_attribution_column(formula: str) -> None:
    event: Final = replace(
        EVENT, user_email=formula, user_id=formula, team_alias=formula, key_alias=formula, version=formula
    )
    content: Final = b"".join([chunk async for chunk in _stream_downloads(event, _events())]).decode()
    row: Final = next(csv.DictReader(io.StringIO(content)))
    assert tuple(row[column] for column in ("user_email", "user_id", "team", "key", "version")) == (f"'{formula}",) * 5


@pytest.mark.asyncio
async def test_empty_history_exports_headers_and_null_attribution_exports_empty_cells() -> None:
    empty: Final = b"".join([chunk async for chunk in _stream_downloads(None, _events())])
    assert empty == b"download_id,downloaded_at,user_email,user_id,team,key,version\r\n"
    event: Final = replace(EVENT, user_email=None, user_id=None, key_alias=None, team_id=None, version=None)
    content: Final = b"".join([chunk async for chunk in _stream_downloads(event, _events())]).decode()
    row: Final = next(csv.DictReader(io.StringIO(content)))
    assert tuple(row[column] for column in ("user_email", "user_id", "team", "key", "version")) == ("",) * 5


@pytest.mark.asyncio
async def test_stream_yields_rows_before_reading_later_events_and_propagates_failures() -> None:
    async def interrupted() -> AsyncIterator[PluginDownloadEvent]:
        yield replace(EVENT, id="2")
        raise RuntimeError("Database unavailable")

    stream: Final = _stream_downloads(EVENT, interrupted())
    assert (await anext(stream)).startswith(b"download_id,")
    assert (await anext(stream)).startswith(EVENT.id.encode() + b",")
    assert (await anext(stream)).startswith(b"2,")
    with pytest.raises(RuntimeError, match="Database unavailable"):
        await anext(stream)
