from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from types import TracebackType
from typing import Final, Protocol

from pydantic import TypeAdapter

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.models.task_insights import (
    UNCLASSIFIED_TASK_ID,
    UNKNOWN_TASK_ID,
    Decision,
    DiscoveredTask,
    DiscoveryRun,
    DraftTaskEdit,
    Freshness,
    InsightsWindow,
    MapPlacement,
    MatrixCell,
    ObservedPoint,
    PendingRequest,
    RequestSample,
    TaskDefinition,
    TaskInsightsState,
    TaskObservation,
    TaskSource,
    TaskVolume,
    UnobservedDelivery,
    WindowTotals,
)


class TaskInsightsExecutor(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...

    async def execute_raw(self, query: str, *args: object) -> int: ...


class _TransactionManager(Protocol):
    async def __aenter__(self) -> TaskInsightsExecutor: ...

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, exc_tb: TracebackType | None
    ) -> None: ...


class TaskInsightsDatabase(TaskInsightsExecutor, Protocol):
    def tx(self, *, timeout: timedelta) -> _TransactionManager: ...


_TIMEOUT: Final = timedelta(seconds=60)
_MAX_EXCERPT_CHARS: Final = 600
_FAILED_ATTEMPT_LIMIT: Final = 3

_UNOBSERVED: Final = """SELECT d.delivery_id, d.request_id, d.is_batch, d.accounting_payload::text AS accounting_payload,
    s.model_id AS model_config_id, s.session_id, s.model AS logged_model,
    COALESCE((
        SELECT jsonb_agg(t.turn ORDER BY t.position)
        FROM jsonb_array_elements(src.turns) WITH ORDINALITY AS t(turn, position)
        WHERE t.position <= 2 OR t.position > jsonb_array_length(src.turns) - 8
    ), '[]'::jsonb)::text AS turns,
    CASE WHEN jsonb_typeof(s.proxy_server_request->'input') = 'string'
        THEN s.proxy_server_request->>'input' END AS input_text
FROM "LiteLLM_SpendLogDelivery" d
LEFT JOIN "LiteLLM_SpendLogs" s ON s.request_id = d.request_id
LEFT JOIN LATERAL (
    SELECT CASE
        WHEN jsonb_typeof(s.proxy_server_request->'messages') = 'array' THEN s.proxy_server_request->'messages'
        WHEN jsonb_typeof(s.proxy_server_request->'input') = 'array' THEN s.proxy_server_request->'input'
        ELSE '[]'::jsonb END AS turns
) src ON true
WHERE d.state = 'applied'
  AND ($2::text IS NULL OR d.request_id LIKE $2::text || '%')
  AND NOT EXISTS (SELECT 1 FROM "LiteLLM_TaskObservation" o WHERE o.delivery_id = d.delivery_id)
ORDER BY d.applied_at, d.delivery_id
LIMIT $1::integer"""

_INSERT_OBSERVATIONS: Final = """INSERT INTO "LiteLLM_TaskObservation" (delivery_id, request_id, is_batch, start_time,
    team_id, model_group, model, model_config_id, session_id, spend, latency_ms, status, request_text, context_text,
    text_hash, excluded)
SELECT delivery_id, request_id, is_batch, start_time, team_id, model_group, model, model_config_id, session_id, spend,
    latency_ms, status, request_text, context_text, text_hash, excluded
FROM jsonb_to_recordset($1::jsonb) AS r(delivery_id text, request_id text, is_batch boolean, start_time timestamp(3),
    team_id text, model_group text, model text, model_config_id text, session_id text, spend double precision,
    latency_ms double precision, status text, request_text text, context_text text, text_hash text, excluded boolean)
ON CONFLICT (delivery_id) DO NOTHING"""

_SCOPED_CTE: Final = (
    """WITH scoped AS (
    SELECT o.* FROM "LiteLLM_TaskObservation" o
    WHERE NOT o.excluded AND o.start_time >= $1::timestamp AND o.start_time < $2::timestamp
      AND ($3::text IS NULL OR o.team_id = $3::text)
),
requests AS (
    SELECT request_id,
        (array_agg(model_group ORDER BY start_time DESC) FILTER (WHERE model_group IS NOT NULL))[1] AS model_group,
        (array_agg(model_config_id ORDER BY start_time DESC) FILTER (WHERE model_config_id IS NOT NULL))[1]
            AS model_config_id,
        (array_agg(session_id ORDER BY start_time DESC) FILTER (WHERE session_id IS NOT NULL))[1] AS session_id,
        (array_agg(request_text ORDER BY start_time DESC) FILTER (WHERE request_text IS NOT NULL))[1] AS request_text,
        MIN(start_time) AS start_time,
        CASE WHEN COUNT(spend) = COUNT(*) THEN SUM(spend) END AS spend,
        MAX(latency_ms) AS latency_ms,
        COUNT(*) AS deliveries,
        bool_or(model_config_id IS NOT NULL) AS attributed
    FROM scoped GROUP BY request_id
),
labeled AS (
    SELECT r.*, a.outcome, a.top_probability,
        CASE WHEN a.outcome = 'assigned' AND d.task_id IS NOT NULL THEN d.task_id
             WHEN a.outcome = 'unknown' THEN '"""
    + UNKNOWN_TASK_ID
    + """'
             ELSE '"""
    + UNCLASSIFIED_TASK_ID
    + """' END AS task_key
    FROM requests r
    LEFT JOIN "LiteLLM_TaskAssignment" a ON a.request_id = r.request_id
    LEFT JOIN "LiteLLM_TaskDefinition" d ON d.task_id = a.task_id AND d.status = 'frozen'
)
"""
)

_VOLUMES: Final = (
    _SCOPED_CTE
    + """SELECT task_key AS task_id, COUNT(*)::bigint AS request_count, SUM(deliveries)::bigint AS delivery_count,
    COUNT(DISTINCT session_id)::bigint AS session_count, SUM(spend)::double precision AS spend,
    COUNT(spend)::bigint AS cost_observations, COUNT(*) FILTER (WHERE NOT attributed)::bigint AS unattributed_count
FROM labeled GROUP BY task_key"""
)

_MATRIX: Final = (
    _SCOPED_CTE
    + """, per_config AS (
    SELECT request_id, model_config_id, CASE WHEN COUNT(spend) = COUNT(*) THEN SUM(spend) END AS spend,
        MAX(latency_ms) AS latency_ms
    FROM scoped WHERE model_config_id IS NOT NULL GROUP BY request_id, model_config_id
)
SELECT c.model_config_id, l.task_key AS task_id, COUNT(*)::bigint AS request_count,
    AVG(c.spend)::double precision AS avg_cost_usd, COUNT(c.spend)::bigint AS cost_observations,
    (percentile_cont(0.95) WITHIN GROUP (ORDER BY c.latency_ms))::double precision AS p95_latency_ms,
    COUNT(c.latency_ms)::bigint AS latency_observations
FROM per_config c JOIN labeled l ON l.request_id = c.request_id
GROUP BY c.model_config_id, l.task_key"""
)

_POINTS: Final = (
    _SCOPED_CTE
    + """SELECT l.request_id, l.task_key AS task_id, p.x, p.y
FROM labeled l
LEFT JOIN "LiteLLM_TaskMapPoint" p ON p.request_id = l.request_id AND p.status = 'frozen'
ORDER BY md5(l.request_id) LIMIT $4::integer"""
)

_SAMPLES: Final = (
    _SCOPED_CTE
    + f"""SELECT request_id, start_time, model_config_id, model_group, session_id, spend, latency_ms,
    deliveries::bigint AS delivery_count, outcome, top_probability, left(request_text, {_MAX_EXCERPT_CHARS}) AS excerpt
FROM labeled WHERE task_key = $4::text
  AND ($5::text IS NULL OR request_id IN (SELECT request_id FROM scoped WHERE model_config_id = $5::text))
ORDER BY start_time DESC, request_id LIMIT $6::integer"""
)

_WINDOW_TOTALS: Final = """SELECT COUNT(DISTINCT request_id) FILTER (WHERE NOT excluded)::bigint AS request_count,
    COUNT(*) FILTER (WHERE NOT excluded)::bigint AS delivery_count,
    COUNT(DISTINCT session_id) FILTER (WHERE NOT excluded)::bigint AS session_count,
    COUNT(DISTINCT request_id) FILTER (WHERE excluded)::bigint AS analysis_request_count,
    (SUM(spend) FILTER (WHERE excluded))::double precision AS analysis_spend
FROM "LiteLLM_TaskObservation"
WHERE start_time >= $1::timestamp AND start_time < $2::timestamp AND ($3::text IS NULL OR team_id = $3::text)"""

_FRESHNESS: Final = """SELECT (SELECT MAX(observed_at) FROM "LiteLLM_TaskObservation") AS last_observed_at,
    (SELECT last_bucketed_at FROM "LiteLLM_TaskInsightsState" WHERE id = 'default') AS last_bucketed_at"""

_STATE: Final = """SELECT discovery_status, discovery_requested_at, discovery_window_start, discovery_window_end,
    discovery_max_requests, discovery_finished_at, discovery_error, discovery_provenance::text AS discovery_provenance,
    taxonomy_frozen_at, last_bucketed_at
FROM "LiteLLM_TaskInsightsState" WHERE id = 'default'"""

_DEFINITION_COLUMNS: Final = """task_id, name, description, inclusion_criteria, exclusion_criteria,
    examples::text AS examples, required_capabilities, status, source, map_x, map_y, discovery_size, created_at,
    frozen_at"""

_DEFINITIONS: Final = f"""SELECT {_DEFINITION_COLUMNS} FROM "LiteLLM_TaskDefinition"
ORDER BY status DESC, discovery_size DESC, name"""

_EDIT_DRAFT: Final = f"""UPDATE "LiteLLM_TaskDefinition" SET name = COALESCE($2, name),
    description = COALESCE($3, description), inclusion_criteria = COALESCE($4, inclusion_criteria),
    exclusion_criteria = COALESCE($5, exclusion_criteria)
WHERE task_id = $1 AND status = 'draft'
RETURNING {_DEFINITION_COLUMNS}"""

_REQUEST_DISCOVERY: Final = """INSERT INTO "LiteLLM_TaskInsightsState" (id, discovery_status, discovery_requested_at,
    discovery_window_start, discovery_window_end, discovery_max_requests, discovery_error, updated_at)
VALUES ('default', 'requested', $1::timestamp, $2::timestamp, $3::timestamp, $4::integer, NULL, $1::timestamp)
ON CONFLICT (id) DO UPDATE SET discovery_status = 'requested', discovery_requested_at = $1::timestamp,
    discovery_window_start = $2::timestamp, discovery_window_end = $3::timestamp,
    discovery_max_requests = $4::integer, discovery_error = NULL, updated_at = $1::timestamp
WHERE "LiteLLM_TaskInsightsState".discovery_status NOT IN ('requested', 'running')
RETURNING id"""

_CLAIM_DISCOVERY: Final = """UPDATE "LiteLLM_TaskInsightsState" SET discovery_status = 'running',
    updated_at = CURRENT_TIMESTAMP
WHERE id = 'default' AND discovery_status = 'requested'
RETURNING discovery_window_start, discovery_window_end, discovery_max_requests"""

_DISCOVERY_SAMPLES: Final = """SELECT request_id, text_hash, request_text, context_text FROM (
    SELECT DISTINCT ON (o.text_hash) o.request_id, o.text_hash, o.request_text, o.context_text
    FROM "LiteLLM_TaskObservation" o
    WHERE NOT o.excluded AND o.request_text IS NOT NULL
      AND o.start_time >= $1::timestamp AND o.start_time < $2::timestamp
    ORDER BY o.text_hash, o.start_time DESC
) sample ORDER BY md5(text_hash) LIMIT $3::integer"""

_FAIL_DISCOVERY: Final = """UPDATE "LiteLLM_TaskInsightsState" SET discovery_status = 'failed',
    discovery_error = left($1, 500), discovery_finished_at = $2::timestamp, updated_at = $2::timestamp
WHERE id = 'default' AND discovery_status IN ('requested', 'running')"""

_FINISH_DISCOVERY: Final = """INSERT INTO "LiteLLM_TaskInsightsState" (id, discovery_status, discovery_finished_at,
    discovery_provenance, updated_at)
VALUES ('default', 'completed', $1::timestamp, $2::jsonb, $1::timestamp)
ON CONFLICT (id) DO UPDATE SET discovery_status = 'completed', discovery_error = NULL,
    discovery_finished_at = $1::timestamp, discovery_provenance = $2::jsonb, updated_at = $1::timestamp"""

_INSERT_DRAFTS: Final = """INSERT INTO "LiteLLM_TaskDefinition" (task_id, name, description, inclusion_criteria,
    exclusion_criteria, examples, required_capabilities, status, source, map_x, map_y, discovery_size, created_at)
SELECT task_id, name, description, inclusion_criteria, exclusion_criteria, examples,
    ARRAY(SELECT jsonb_array_elements_text(required_capabilities)), 'draft', $3::text, map_x, map_y,
    discovery_size, $2::timestamp
FROM jsonb_to_recordset($1::jsonb) AS r(task_id text, name text, description text, inclusion_criteria text,
    exclusion_criteria text, examples jsonb, required_capabilities jsonb, map_x double precision,
    map_y double precision, discovery_size integer)"""

_INSERT_DRAFT_POINTS: Final = """INSERT INTO "LiteLLM_TaskMapPoint" (request_id, status, task_id, x, y)
SELECT request_id, 'draft', task_id, x, y
FROM jsonb_to_recordset($1::jsonb) AS r(request_id text, task_id text, x double precision, y double precision)
ON CONFLICT (request_id, status) DO UPDATE SET task_id = EXCLUDED.task_id, x = EXCLUDED.x, y = EXCLUDED.y"""

_LOCK_STATE: Final = """INSERT INTO "LiteLLM_TaskInsightsState" (id) VALUES ('default')
ON CONFLICT (id) DO UPDATE SET updated_at = "LiteLLM_TaskInsightsState".updated_at"""
_TAXONOMY_VERSION: Final = """SELECT taxonomy_frozen_at FROM "LiteLLM_TaskInsightsState" WHERE id = 'default'"""
_LOCK_CURRENT_TAXONOMY: Final = """SELECT 1 AS current FROM "LiteLLM_TaskInsightsState"
WHERE id = 'default' AND taxonomy_frozen_at IS NOT DISTINCT FROM $1::timestamp FOR UPDATE"""
_FREEZE: Final = (
    """DELETE FROM "LiteLLM_TaskDefinition" WHERE status = 'frozen' AND EXISTS (
    SELECT 1 FROM "LiteLLM_TaskDefinition" WHERE status = 'draft')""",
    """DELETE FROM "LiteLLM_TaskMapPoint" WHERE status = 'frozen' AND EXISTS (
    SELECT 1 FROM "LiteLLM_TaskDefinition" WHERE status = 'draft')""",
    """UPDATE "LiteLLM_TaskMapPoint" SET status = 'frozen' WHERE status = 'draft'""",
    """DELETE FROM "LiteLLM_TaskAssignment" WHERE EXISTS (
    SELECT 1 FROM "LiteLLM_TaskDefinition" WHERE status = 'draft')""",
)
_FREEZE_DEFINITIONS: Final = """UPDATE "LiteLLM_TaskDefinition" SET status = 'frozen', frozen_at = $1::timestamp
WHERE status = 'draft'"""
_MARK_FROZEN: Final = """INSERT INTO "LiteLLM_TaskInsightsState" (id, taxonomy_frozen_at, updated_at)
VALUES ('default', $1::timestamp, $1::timestamp)
ON CONFLICT (id) DO UPDATE SET taxonomy_frozen_at = $1::timestamp, updated_at = $1::timestamp"""

_PENDING: Final = f"""SELECT DISTINCT ON (o.request_id) o.request_id, o.text_hash, o.request_text, o.context_text
FROM "LiteLLM_TaskObservation" o
WHERE NOT o.excluded AND o.request_text IS NOT NULL AND o.text_hash IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM "LiteLLM_TaskAssignment" a
      WHERE a.request_id = o.request_id AND (a.outcome <> 'failed' OR a.attempts >= {_FAILED_ATTEMPT_LIMIT}))
ORDER BY o.request_id, o.start_time DESC
LIMIT $1::integer"""

_BY_HASH: Final = """SELECT DISTINCT ON (text_hash) text_hash, outcome, task_id, probabilities::text AS probabilities,
    top_probability, fit_probability, decision_model, decision_provenance::text AS provenance
FROM "LiteLLM_TaskAssignment"
WHERE text_hash = ANY($1::text[]) AND outcome <> 'failed'
ORDER BY text_hash, assigned_at DESC"""

_SAVE_DECISIONS: Final = """INSERT INTO "LiteLLM_TaskAssignment" (request_id, task_id, outcome, text_hash,
    probabilities, top_probability, fit_probability, decision_model, decision_provenance, attempts, error, assigned_at)
SELECT request_id, task_id, outcome, text_hash, probabilities, top_probability, fit_probability, decision_model,
    provenance, CASE WHEN outcome = 'failed' THEN 1 ELSE 0 END, error, $2::timestamp
FROM jsonb_to_recordset($1::jsonb) AS r(request_id text, task_id text, outcome text, text_hash text,
    probabilities jsonb, top_probability double precision, fit_probability double precision, decision_model text,
    provenance jsonb, error text)
ON CONFLICT (request_id) DO UPDATE SET task_id = EXCLUDED.task_id, outcome = EXCLUDED.outcome,
    text_hash = EXCLUDED.text_hash, probabilities = EXCLUDED.probabilities,
    top_probability = EXCLUDED.top_probability, fit_probability = EXCLUDED.fit_probability,
    decision_model = EXCLUDED.decision_model, decision_provenance = EXCLUDED.decision_provenance,
    attempts = "LiteLLM_TaskAssignment".attempts + EXCLUDED.attempts, error = EXCLUDED.error,
    assigned_at = EXCLUDED.assigned_at"""

_MARK_BUCKETED: Final = """INSERT INTO "LiteLLM_TaskInsightsState" (id, last_bucketed_at, updated_at)
VALUES ('default', $1::timestamp, $1::timestamp)
ON CONFLICT (id) DO UPDATE SET last_bucketed_at = $1::timestamp, updated_at = $1::timestamp"""

_RECOVER_DISCOVERY: Final = """UPDATE "LiteLLM_TaskInsightsState" SET discovery_status = 'failed',
    discovery_error = 'The analysis worker restarted during discovery', updated_at = CURRENT_TIMESTAMP
WHERE id = 'default' AND discovery_status = 'running'"""


_UNOBSERVED_ROWS: Final = TypeAdapter(tuple[UnobservedDelivery, ...])
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])
_TURNS: Final = TypeAdapter(tuple[object, ...])
_STATES: Final = TypeAdapter(tuple[TaskInsightsState, ...])
_DEFINITIONS_ADAPTER: Final = TypeAdapter(tuple[TaskDefinition, ...])
_VOLUMES_ADAPTER: Final = TypeAdapter(tuple[TaskVolume, ...])
_CELLS_ADAPTER: Final = TypeAdapter(tuple[MatrixCell, ...])
_POINTS_ADAPTER: Final = TypeAdapter(tuple[ObservedPoint, ...])
_SAMPLES_ADAPTER: Final = TypeAdapter(tuple[RequestSample, ...])
_PENDING_ADAPTER: Final = TypeAdapter(tuple[PendingRequest, ...])
_TOTALS_ADAPTER: Final = TypeAdapter(tuple[WindowTotals, ...])
_FRESHNESS_ADAPTER: Final = TypeAdapter(tuple[Freshness, ...])
_EXAMPLES: Final = TypeAdapter(tuple[str, ...])


def _db_time(value: datetime) -> str:
    utc: Final = value.astimezone(timezone.utc) if value.tzinfo is not None else value
    return utc.replace(tzinfo=None).isoformat(timespec="milliseconds")


def _parsed_json(value: object) -> object:
    return _JSON_OBJECT.validate_json(value) if isinstance(value, str) else value


def _unobserved(row: Mapping[str, object]) -> Mapping[str, object]:
    turns: Final = row.get("turns")
    return {
        **row,
        "accounting_payload": _parsed_json(row.get("accounting_payload")),
        "turns": _TURNS.validate_json(turns) if isinstance(turns, str) else (),
    }


def _definition(row: Mapping[str, object]) -> Mapping[str, object]:
    examples: Final = row.get("examples")
    return {**row, "examples": _EXAMPLES.validate_json(examples) if isinstance(examples, str) else ()}


def _state(row: Mapping[str, object]) -> Mapping[str, object]:
    return {**row, "discovery_provenance": _parsed_json(row.get("discovery_provenance")) or {}}


def _observation_record(observation: TaskObservation) -> Mapping[str, object]:
    return {**observation.model_dump(mode="json"), "start_time": _db_time(observation.start_time)}


def _decision_record(pending: PendingRequest, decision: Decision) -> Mapping[str, object]:
    return {
        "request_id": pending.request_id,
        "task_id": decision.task_id,
        "outcome": decision.outcome,
        "text_hash": pending.text_hash,
        "probabilities": dict(decision.probabilities),
        "top_probability": decision.top_probability,
        "fit_probability": decision.fit_probability,
        "decision_model": decision.decision_model,
        "provenance": dict(decision.provenance),
        "error": decision.error,
    }


def _decision_from_row(row: Mapping[str, object]) -> Decision:
    return Decision.model_validate(
        {
            **row,
            "probabilities": _parsed_json(row.get("probabilities")) or {},
            "provenance": {
                **dict(_JSON_OBJECT.validate_python(_parsed_json(row.get("provenance")) or {})),
                "reused": 1,
            },
        }
    )


def _window_args(window: InsightsWindow) -> tuple[str, str, str | None]:
    return _db_time(window.start), _db_time(window.end), window.team_id


class TaskInsightsRepository:
    """Task Insights persistence: the observation projection of applied ledger deliveries, the shared taxonomy,
    per-request assignments and the aggregates read from them."""

    def __init__(self, database: TaskInsightsDatabase) -> None:
        self._database = database

    async def unobserved_deliveries(
        self, limit: int, request_id_prefix: str | None = None
    ) -> tuple[UnobservedDelivery, ...]:
        rows: Final = await self._database.query_raw(_UNOBSERVED, limit, request_id_prefix)
        return _UNOBSERVED_ROWS.validate_python(tuple(_unobserved(row) for row in rows))

    async def insert_observations(self, observations: tuple[TaskObservation, ...]) -> int:
        if not observations:
            return 0
        return await self._database.execute_raw(
            _INSERT_OBSERVATIONS,
            safe_dumps(tuple(_observation_record(item) for item in observations), non_finite_as_null=True),
        )

    async def get_state(self) -> TaskInsightsState:
        states: Final = _STATES.validate_python(tuple(_state(row) for row in await self._database.query_raw(_STATE)))
        return states[0] if states else TaskInsightsState()

    async def list_definitions(self) -> tuple[TaskDefinition, ...]:
        rows: Final = await self._database.query_raw(_DEFINITIONS)
        return _DEFINITIONS_ADAPTER.validate_python(tuple(_definition(row) for row in rows))

    async def frozen_definitions(self) -> tuple[TaskDefinition, ...]:
        return tuple(task for task in await self.list_definitions() if task.status == "frozen")

    async def task_volumes(self, window: InsightsWindow) -> tuple[TaskVolume, ...]:
        return _VOLUMES_ADAPTER.validate_python(await self._database.query_raw(_VOLUMES, *_window_args(window)))

    async def matrix_cells(self, window: InsightsWindow) -> tuple[MatrixCell, ...]:
        return _CELLS_ADAPTER.validate_python(await self._database.query_raw(_MATRIX, *_window_args(window)))

    async def map_points(self, window: InsightsWindow, limit: int) -> tuple[ObservedPoint, ...]:
        return _POINTS_ADAPTER.validate_python(await self._database.query_raw(_POINTS, *_window_args(window), limit))

    async def request_samples(
        self, window: InsightsWindow, task_id: str, model_config_id: str | None, limit: int
    ) -> tuple[RequestSample, ...]:
        rows: Final = await self._database.query_raw(_SAMPLES, *_window_args(window), task_id, model_config_id, limit)
        return _SAMPLES_ADAPTER.validate_python(rows)

    async def window_totals(self, window: InsightsWindow) -> WindowTotals:
        rows: Final = _TOTALS_ADAPTER.validate_python(
            await self._database.query_raw(_WINDOW_TOTALS, *_window_args(window))
        )
        return rows[0]

    async def freshness(self) -> Freshness:
        rows: Final = _FRESHNESS_ADAPTER.validate_python(await self._database.query_raw(_FRESHNESS))
        return rows[0] if rows else Freshness(last_observed_at=None, last_bucketed_at=None)

    async def edit_draft(self, task_id: str, edit: DraftTaskEdit) -> TaskDefinition | None:
        rows: Final = await self._database.query_raw(
            _EDIT_DRAFT, task_id, edit.name, edit.description, edit.inclusion_criteria, edit.exclusion_criteria
        )
        updated: Final = _DEFINITIONS_ADAPTER.validate_python(tuple(_definition(row) for row in rows))
        return updated[0] if updated else None

    async def freeze_draft(self, frozen_at: datetime) -> int:
        """Replacing the taxonomy drops every assignment so the worker buckets all requests against the new one."""
        async with self._database.tx(timeout=_TIMEOUT) as transaction:
            await transaction.execute_raw(_LOCK_STATE)
            for statement in _FREEZE:
                await transaction.execute_raw(statement)
            frozen: Final = await transaction.execute_raw(_FREEZE_DEFINITIONS, _db_time(frozen_at))
            if frozen:
                await transaction.execute_raw(_MARK_FROZEN, _db_time(frozen_at))
        return frozen

    async def request_discovery(self, window: InsightsWindow, max_requests: int, requested_at: datetime) -> bool:
        rows: Final = await self._database.query_raw(
            _REQUEST_DISCOVERY, _db_time(requested_at), _db_time(window.start), _db_time(window.end), max_requests
        )
        return bool(rows)

    async def recover_discovery(self) -> int:
        return await self._database.execute_raw(_RECOVER_DISCOVERY)

    async def claim_discovery(self) -> DiscoveryRun | None:
        claims: Final = _STATES.validate_python(await self._database.query_raw(_CLAIM_DISCOVERY))
        if not claims:
            return None
        claim: Final = claims[0]
        if claim.discovery_window_start is None or claim.discovery_window_end is None:
            return None
        return DiscoveryRun(
            window=(claim.discovery_window_start, claim.discovery_window_end), max_requests=claim.discovery_max_requests
        )

    async def discovery_samples(self, run: DiscoveryRun) -> tuple[PendingRequest, ...]:
        start, end = run.window
        rows: Final = await self._database.query_raw(
            _DISCOVERY_SAMPLES, _db_time(start), _db_time(end), run.max_requests
        )
        return _PENDING_ADAPTER.validate_python(rows)

    async def save_draft_taxonomy(
        self,
        tasks: tuple[DiscoveredTask, ...],
        placements: tuple[MapPlacement, ...],
        provenance: Mapping[str, str | int | float | None],
        finished_at: datetime,
        source: TaskSource = "discovered",
    ) -> None:
        async with self._database.tx(timeout=_TIMEOUT) as transaction:
            await transaction.execute_raw("""DELETE FROM "LiteLLM_TaskDefinition" WHERE status = 'draft'""")
            await transaction.execute_raw("""DELETE FROM "LiteLLM_TaskMapPoint" WHERE status = 'draft'""")
            if tasks:
                await transaction.execute_raw(
                    _INSERT_DRAFTS,
                    safe_dumps(tuple(task.model_dump(mode="json") for task in tasks), non_finite_as_null=True),
                    _db_time(finished_at),
                    source,
                )
            if placements:
                await transaction.execute_raw(
                    _INSERT_DRAFT_POINTS,
                    safe_dumps(tuple(point.model_dump(mode="json") for point in placements), non_finite_as_null=True),
                )
            await transaction.execute_raw(_FINISH_DISCOVERY, _db_time(finished_at), safe_dumps(dict(provenance)))

    async def fail_discovery(self, error: str, finished_at: datetime) -> None:
        await self._database.execute_raw(_FAIL_DISCOVERY, error, _db_time(finished_at))

    async def pending_requests(self, limit: int) -> tuple[PendingRequest, ...]:
        return _PENDING_ADAPTER.validate_python(await self._database.query_raw(_PENDING, limit))

    async def decisions_by_hash(self, text_hashes: tuple[str, ...]) -> Mapping[str, Decision]:
        if not text_hashes:
            return {}
        rows: Final = await self._database.query_raw(_BY_HASH, list(text_hashes))
        return {str(row["text_hash"]): _decision_from_row(row) for row in rows}

    async def taxonomy_version(self) -> datetime | None:
        rows: Final = _STATES.validate_python(await self._database.query_raw(_TAXONOMY_VERSION))
        return rows[0].taxonomy_frozen_at if rows else None

    async def save_decisions(
        self,
        decisions: tuple[tuple[PendingRequest, Decision], ...],
        bucketed_at: datetime,
        taxonomy_version: datetime | None,
    ) -> int:
        """Writes nothing when the taxonomy was replaced after the decisions were made. Freezing locks the same state
        row first, so a freeze and a save never interleave."""
        if not decisions:
            return 0
        async with self._database.tx(timeout=_TIMEOUT) as transaction:
            current: Final = await transaction.query_raw(
                _LOCK_CURRENT_TAXONOMY, _db_time(taxonomy_version) if taxonomy_version is not None else None
            )
            if not current:
                return 0
            saved: Final = await transaction.execute_raw(
                _SAVE_DECISIONS,
                safe_dumps(tuple(_decision_record(*item) for item in decisions), non_finite_as_null=True),
                _db_time(bucketed_at),
            )
            await transaction.execute_raw(_MARK_BUCKETED, _db_time(bucketed_at))
        return saved
