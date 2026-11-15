from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock

from litellm.models.alert import Alert, AlertDetection
from litellm.repositories.alert_repository import AlertRepository

_START: Final = datetime(2026, 1, 8, tzinfo=timezone.utc)
_DETECTION: Final = AlertDetection.model_validate(
    {
        "type": "prompt_cache",
        "resource_type": "deployment",
        "resource_id": "claude-east",
        "window_start": _START,
        "window_end": _START + timedelta(seconds=3),
        "evidence": {
            "config_fingerprint": "config-1",
            "prefix_fingerprint": "prefix-1",
            "provider": "anthropic",
            "provider_model": "claude-test",
            "consecutive_misses": 3,
            "proven_write_tokens": 2048,
            "ttl_seconds": 300,
        },
    }
)


class _Row:
    def __init__(self, values: Mapping[str, object]) -> None:
        self._values: Final = dict(values)

    def dict(self) -> dict[str, object]:
        return self._values


def _row(status: str) -> _Row:
    return _Row(
        {
            **_DETECTION.model_dump(),
            "id": 7,
            "detection_key": "key",
            "status": status,
            "created_at": _START,
            "updated_at": _START,
            "updated_by": None,
        }
    )


class _Transaction:
    def __init__(self) -> None:
        self.queries: tuple[str, ...] = ()

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        self.queries = (*self.queries, query)
        return ()


def _repository(transaction: _Transaction) -> tuple[AlertRepository, MagicMock]:
    client: Final = MagicMock()

    @asynccontextmanager
    async def tx(*, timeout: timedelta) -> AsyncIterator[_Transaction]:
        yield transaction

    client.writer_db.tx = tx
    return AlertRepository(client), client.writer_db.litellm_alert


async def test_a_later_detection_of_a_dismissed_incident_returns_it_unchanged() -> None:
    """A dismissed incident stays the only one for its deployment and configuration, even when a later window
    and another prefix find the same problem."""
    from prisma.errors import UniqueViolationError

    later: Final = _DETECTION.model_copy(
        update={
            "window_start": _START + timedelta(hours=2),
            "window_end": _START + timedelta(hours=2, seconds=3),
            "evidence": _DETECTION.evidence.model_copy(update={"prefix_fingerprint": "prefix-2"}),
        }
    )
    repository, table = _repository(_Transaction())
    table.create = AsyncMock(side_effect=UniqueViolationError({}, message="detection_key"))
    table.find_unique = AsyncMock(return_value=_row("dismissed"))
    await repository.record(_DETECTION)
    saved: Final = await repository.record(later)
    first_key, later_key = (call.kwargs["data"]["detection_key"] for call in table.create.await_args_list)
    assert first_key == later_key
    assert saved.status == "dismissed"
    table.update.assert_not_called()


async def test_prompt_cache_alerts_are_returned_without_looking_up_metric_history() -> None:
    transaction: Final = _Transaction()
    repository, _ = _repository(transaction)
    incident: Final = Alert.model_validate(
        {**_DETECTION.model_dump(), "id": "7", "status": "open", "created_at": _START, "updated_at": _START}
    )
    (view,) = await repository.enrich((incident,))
    assert transaction.queries == ()
    assert (view.id, view.history) == ("7", None)
