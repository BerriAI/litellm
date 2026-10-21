from datetime import datetime
from typing import Final, Protocol

from pydantic import TypeAdapter

from litellm.proxy.lens.models import Dataset, DatasetSummary, Record
from litellm.proxy.lens.repository import Database, Row

_ROWS: Final = TypeAdapter(tuple[Row, ...])


class StoredSummary(Record):
    team_id: str
    summary: DatasetSummary


class DatasetStore(Protocol):
    async def summaries(self) -> tuple[StoredSummary, ...]: ...
    async def get(self, dataset_id: str, revision: int | None = None) -> Dataset | None: ...
    async def insert(self, dataset: Dataset, saved_at: datetime) -> bool: ...


class DatasetRepository:
    def __init__(self, db: Database) -> None:
        self.db: Final = db

    async def summaries(self) -> tuple[StoredSummary, ...]:
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                """SELECT jsonb_build_object(
                    'team_id', data->>'team_id',
                    'summary', jsonb_build_object(
                        'id', id, 'name', data->>'name', 'agent_name', data->>'agent_name', 'revision', revision,
                        'case_count', jsonb_array_length(data->'cases'),
                        'updated_at', created_at AT TIME ZONE 'UTC'
                    )
                ) AS data FROM (
                    SELECT DISTINCT ON (id) id, revision, created_at, data FROM "LiteLLM_LensDataset"
                    ORDER BY id, revision DESC
                ) AS latest ORDER BY created_at DESC"""
            )
        )
        return tuple(StoredSummary.model_validate(row.data) for row in rows)

    async def get(self, dataset_id: str, revision: int | None = None) -> Dataset | None:
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                """SELECT data FROM "LiteLLM_LensDataset" WHERE id=$1 AND ($2::int IS NULL OR revision=$2::int)
                ORDER BY revision DESC LIMIT 1""",
                dataset_id,
                revision,
            )
        )
        return Dataset.model_validate(rows[0].data) if rows else None

    async def insert(self, dataset: Dataset, saved_at: datetime) -> bool:
        inserted: Final = await self.db.execute_raw(
            """INSERT INTO "LiteLLM_LensDataset" (id, revision, created_at, data)
            VALUES ($1, $2, $3::timestamptz AT TIME ZONE 'UTC', $4::jsonb) ON CONFLICT (id, revision) DO NOTHING""",
            dataset.id,
            dataset.revision,
            saved_at.isoformat(),
            dataset.model_dump_json(),
        )
        return inserted == 1
