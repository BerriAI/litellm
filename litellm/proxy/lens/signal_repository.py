import json
from datetime import datetime
from typing import Final

from pydantic import TypeAdapter

from litellm.proxy.lens.models import Execution, TraceIdentity
from litellm.proxy.lens.repository import Database, Row
from litellm.proxy.lens.signals import (
    SIGNAL_RECLASSIFY_AFTER,
    SIGNAL_RETRY_FAILED_AFTER,
    SignalAttempt,
    SignalConfig,
    StoredTraceSignal,
)

_ROWS: Final[TypeAdapter[tuple[Row, ...]]] = TypeAdapter(tuple[Row, ...])


class SignalRepository:
    def __init__(self, db: Database) -> None:
        self.db: Final = db

    async def get_config(self) -> SignalConfig:
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw('SELECT data FROM "LiteLLM_LensSignalConfig" WHERE id=$1', "global")
        )
        return SignalConfig() if not rows else SignalConfig.model_validate(rows[0].data)

    async def save_config(self, config: SignalConfig) -> None:
        await self.db.execute_raw(
            """INSERT INTO "LiteLLM_LensSignalConfig" (id, data)
            VALUES ($1, $2::jsonb)
            ON CONFLICT (id) DO UPDATE SET data=EXCLUDED.data""",
            "global",
            json.dumps(config.model_dump(mode="json")),
        )

    async def traces(self, identities: tuple[TraceIdentity, ...]) -> tuple[StoredTraceSignal, ...]:
        if not identities:
            return ()
        payload: Final = json.dumps(
            tuple({"trace_id": trace.trace_id, "trace_ref": trace.trace_ref} for trace in identities)
        )
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                """SELECT jsonb_build_object(
                    'trace_id', trace_id,
                    'trace_ref', trace_ref,
                    'config_key', config_key,
                    'span_count', span_count,
                    'claimed_until', claimed_until,
                    'classified_at', classified_at,
                    'data', data
                ) AS data
                FROM "LiteLLM_LensTraceSignal"
                WHERE (trace_id, trace_ref) IN (
                    SELECT trace_id, trace_ref FROM jsonb_to_recordset($1::jsonb) AS requested(
                        trace_id text, trace_ref text
                    )
                )""",
                payload,
            )
        )
        return tuple(StoredTraceSignal.model_validate(row.data) for row in rows)

    async def claim(
        self,
        execution: Execution,
        config: SignalConfig,
        claimed_until: datetime,
        now: datetime,
    ) -> bool:
        data: Final = json.dumps({"status": "pending", "scores": {}, "model": config.model, "error": ""})
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                """INSERT INTO "LiteLLM_LensTraceSignal" AS stored
                    (trace_id, trace_ref, config_key, span_count, claimed_until, classified_at, data)
                VALUES ($1, $2, $3, $4, $5::timestamp, NULL, $6::jsonb)
                ON CONFLICT (trace_id, trace_ref) DO UPDATE SET
                    config_key=EXCLUDED.config_key,
                    span_count=EXCLUDED.span_count,
                    claimed_until=EXCLUDED.claimed_until,
                    classified_at=NULL,
                    data=EXCLUDED.data
                WHERE (stored.claimed_until IS NULL OR stored.claimed_until < $7::timestamp)
                    AND (
                        stored.config_key IS DISTINCT FROM EXCLUDED.config_key
                        OR (
                            stored.data->>'status'='pending'
                            AND stored.claimed_until < $7::timestamp
                        )
                        OR (
                            EXCLUDED.span_count > stored.span_count
                            AND stored.classified_at < $8::timestamp
                        )
                        OR (
                            stored.data->>'status'='failed'
                            AND stored.classified_at < $9::timestamp
                        )
                    )
                RETURNING jsonb_build_object('trace_id', trace_id) AS data""",
                execution.trace_id,
                execution.trace_ref,
                config.key(),
                execution.span_count,
                claimed_until,
                data,
                now,
                now - SIGNAL_RECLASSIFY_AFTER,
                now - SIGNAL_RETRY_FAILED_AFTER,
            )
        )
        return bool(rows)

    async def store(
        self,
        execution: Execution,
        config: SignalConfig,
        claimed_until: datetime,
        classified_at: datetime,
        attempt: SignalAttempt,
    ) -> None:
        payload: Final = json.dumps(
            {
                "status": attempt.status,
                "scores": dict(attempt.scores),
                "model": attempt.model,
                "error": attempt.error,
            }
        )
        await self.db.execute_raw(
            """UPDATE "LiteLLM_LensTraceSignal"
            SET classified_at=$1::timestamp, claimed_until=NULL, data=$2::jsonb
            WHERE trace_id=$3 AND trace_ref=$4 AND config_key=$5 AND claimed_until=$6::timestamp""",
            classified_at,
            payload,
            execution.trace_id,
            execution.trace_ref,
            config.key(),
            claimed_until,
        )
