from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.integrations.focus.destinations.base import FocusTimeWindow
from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import MavvrikFocusLogger


class _Frame:
    def __init__(self, *, empty: bool) -> None:
        self._empty = empty

    def __len__(self) -> int:
        return 0 if self._empty else 1

    def is_empty(self) -> bool:
        return self._empty


def _window() -> FocusTimeWindow:
    return FocusTimeWindow(
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
        frequency="daily",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data_empty", "normalized_empty", "serialized_payload"),
    (
        (True, False, b"not-used"),
        (False, True, b"not-used"),
        (False, False, b""),
    ),
)
async def test_export_window_delivers_empty_payload_for_empty_export(
    data_empty: bool,
    normalized_empty: bool,
    serialized_payload: bytes,
) -> None:
    database = MagicMock()
    database.get_usage_data = AsyncMock(return_value=_Frame(empty=data_empty))
    transformer = MagicMock()
    transformer.transform.return_value = _Frame(empty=normalized_empty)
    serializer = MagicMock()
    serializer.serialize.return_value = serialized_payload
    destination = MagicMock()
    destination.deliver = AsyncMock()
    engine = MagicMock()
    engine._database = database
    engine._transformer = transformer
    engine._serializer = serializer
    engine._destination = destination
    engine._build_filename.return_value = "metrics.csv"
    logger = MavvrikFocusLogger()
    logger._engine = engine
    window = _window()

    await logger._export_window(window=window, limit=None)

    destination.deliver.assert_awaited_once_with(
        content=b"",
        time_window=window,
        filename="metrics.csv",
    )
