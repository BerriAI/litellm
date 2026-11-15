from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock

from litellm.models.alert import METRIC_ALERT_TYPES, Alert, AlertDetection
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
        self._values = dict(values)

    def dict(self) -> dict[str, object]:
        return self._values


def _row(detection: AlertDetection, status: str) -> _Row:
    return _Row(
        {
            **detection.model_dump(),
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
        self.queries: tuple[tuple[str, tuple[object, ...]], ...] = ()

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        self.queries = (*self.queries, (query, args))
        return ()


def _repository(transaction: _Transaction | None = None) -> tuple[AlertRepository, MagicMock]:
    client = MagicMock()
    opened = transaction or _Transaction()

    @asynccontextmanager
    async def tx(*, timeout: timedelta) -> AsyncIterator[_Transaction]:
        yield opened

    client.writer_db.tx = tx
    return AlertRepository(client), client.writer_db.litellm_alert


async def test_the_insert_that_wins_the_detection_key_is_the_only_one_reported_as_created() -> None:
    repository, table = _repository()
    table.create = AsyncMock(return_value=_row(_DETECTION, "open"))
    recorded = await repository.record_once(_DETECTION)
    assert (recorded.created, recorded.alert.status) == (True, "open")


async def test_a_detection_of_a_dismissed_incident_returns_it_unchanged_and_not_created() -> None:
    from prisma.errors import UniqueViolationError

    repository, table = _repository()
    table.create = AsyncMock(side_effect=UniqueViolationError({}, message="detection_key"))
    table.find_unique = AsyncMock(return_value=_row(_DETECTION, "dismissed"))
    recorded = await repository.record_once(_DETECTION)
    assert (recorded.created, recorded.alert.status) == (False, "dismissed")
    table.update.assert_not_called()


async def test_performance_listing_names_the_metric_types_so_new_categories_stay_out() -> None:
    repository, table = _repository()
    table.find_many = AsyncMock(return_value=[])
    await repository.list("open", 10, None, "performance")
    where = table.find_many.await_args.kwargs["where"]
    assert where["type"] == {"in": list(METRIC_ALERT_TYPES)}
    assert "prompt_cache" not in where["type"]["in"] and "model_deprecation" not in where["type"]["in"]


async def test_daily_summaries_only_read_metric_alerts() -> None:
    transaction = _Transaction()
    repository, _ = _repository(transaction)
    await repository.summaries(0, 10)
    (_, args) = transaction.queries[-1]
    assert args[-1] == list(METRIC_ALERT_TYPES)


async def test_history_is_only_looked_up_for_metric_alerts() -> None:
    transaction = _Transaction()
    repository, _ = _repository(transaction)
    incident = Alert.model_validate(_row(_DETECTION, "open").dict() | {"id": "7"})
    views = await repository.enrich((incident,))
    assert transaction.queries == ()
    assert views[0].id == "7" and views[0].history is None


async def test_a_mixed_page_looks_up_history_only_for_its_metric_resources() -> None:
    transaction = _Transaction()
    repository, _ = _repository(transaction)
    incident = Alert.model_validate(_row(_DETECTION, "open").dict() | {"id": "7"})
    spend = Alert.model_validate(
        {
            "id": "8",
            "type": "spend",
            "resource_type": "team",
            "resource_id": "team-1",
            "status": "open",
            "window_start": _START,
            "window_end": _START + timedelta(days=1),
            "evidence": {"value": 20, "baseline": 2, "unit": "usd", "sample_size": 50},
            "created_at": _START,
            "updated_at": _START,
        }
    )
    views = await repository.enrich((incident, spend))
    (query, _) = transaction.queries[-1]
    assert [view.id for view in views] == ["7", "8"]
    assert '"LiteLLM_DailyTeamSpend"' in query and "deployment" not in query
