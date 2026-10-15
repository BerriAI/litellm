from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final, Literal

import psycopg
from psycopg import sql
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter

TARGET_BYTES: Final = 1_000_000_000_000
BATCH_ROWS: Final = 100_000
SEED_LOCK: Final = 692834012
INTEGER: Final = TypeAdapter(int)
TEXT: Final = TypeAdapter(str)


class SeedSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    baseline_version: str = Field(pattern=r"^v\d+\.\d+\.\d+$")
    target_bytes: int = Field(default=TARGET_BYTES, gt=0)
    anchor: AwareDatetime
    profile: Literal["metadata-only-v1"] = "metadata-only-v1"


class SeedProgress(BaseModel):
    model_config = ConfigDict(frozen=True)

    rows: int
    table_bytes: int
    index_bytes: int
    total_bytes: int


class SeedArguments(SeedSpec):
    batch_rows: int = Field(default=BATCH_ROWS, gt=0, le=1_000_000)
    confirm_disposable: Literal[True]
    manifest: Path | None = None


class SeedManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    spec: SeedSpec
    progress: SeedProgress
    heap_bytes: int
    toast_bytes: int
    database_bytes: int
    postgres_version: str
    migration_hash: str
    generator_hash: str
    indexes: tuple[str, ...]
    store_prompts_in_spend_logs: Literal[False] = False


def generator_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_suffix(".sql").read_bytes()).hexdigest()


def save_manifest(connection: psycopg.Connection[tuple[object, ...]], spec: SeedSpec) -> SeedManifest:
    record: Final = connection.execute(
        "SELECT pg_relation_size('\"LiteLLM_SpendLogs\"'), "
        "CASE WHEN reltoastrelid = 0 THEN 0 ELSE pg_total_relation_size(reltoastrelid) END, "
        "pg_database_size(current_database()), current_setting('server_version'), migration_hash, generator_hash "
        "FROM pg_class CROSS JOIN litellm_perf_seed.checkpoint WHERE oid = '\"LiteLLM_SpendLogs\"'::regclass"
    ).fetchone()
    assert record is not None
    result: Final = SeedManifest(
        spec=spec,
        progress=checkpoint_progress(connection),
        heap_bytes=INTEGER.validate_python(record[0]),
        toast_bytes=INTEGER.validate_python(record[1]),
        database_bytes=INTEGER.validate_python(record[2]),
        postgres_version=TEXT.validate_python(record[3]),
        migration_hash=TEXT.validate_python(record[4]),
        generator_hash=TEXT.validate_python(record[5]),
        indexes=tuple(
            TEXT.validate_python(row[0])
            for row in connection.execute(
                "SELECT indexdef FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'LiteLLM_SpendLogs' "
                "ORDER BY indexname"
            )
        ),
    )
    connection.execute("UPDATE litellm_perf_seed.checkpoint SET manifest = %s::jsonb", (result.model_dump_json(),))
    return result


def scalar(connection: psycopg.Connection[tuple[object, ...]], query: bytes, params: tuple[object, ...] = ()) -> object:
    row: Final = connection.execute(query, params or None).fetchone()
    assert row is not None, "Seed query returned no row"
    return row[0]


def measure(connection: psycopg.Connection[tuple[object, ...]], rows: int) -> SeedProgress:
    sizes: Final = connection.execute(
        "SELECT pg_table_size('\"LiteLLM_SpendLogs\"'), pg_indexes_size('\"LiteLLM_SpendLogs\"'), "
        "pg_total_relation_size('\"LiteLLM_SpendLogs\"')"
    ).fetchone()
    assert sizes is not None
    return SeedProgress(
        rows=rows,
        table_bytes=INTEGER.validate_python(sizes[0]),
        index_bytes=INTEGER.validate_python(sizes[1]),
        total_bytes=INTEGER.validate_python(sizes[2]),
    )


def initialize(connection: psycopg.Connection[tuple[object, ...]], spec: SeedSpec) -> None:
    assert scalar(connection, b"SELECT relkind FROM pg_class WHERE oid = '\"LiteLLM_SpendLogs\"'::regclass") == "r", (
        "This seed profile requires unpartitioned spend logs"
    )
    with connection.transaction():
        connection.execute("CREATE SCHEMA IF NOT EXISTS litellm_perf_seed")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS litellm_perf_seed.checkpoint "
            "(id boolean PRIMARY KEY DEFAULT true CHECK(id), spec jsonb NOT NULL, "
            "rows bigint NOT NULL DEFAULT 0, migration_hash text NOT NULL, generator_hash text NOT NULL, manifest jsonb)"
        )
        clean: Final = scalar(
            connection,
            b'SELECT count(*) FROM "_prisma_migrations" WHERE finished_at IS NULL AND rolled_back_at IS NULL',
        )
        assert clean == 0, "Baseline has unfinished migrations"
        history: Final = TEXT.validate_python(
            scalar(
                connection,
                b"SELECT md5(string_agg(migration_name || checksum, ',' ORDER BY migration_name)) "
                b'FROM "_prisma_migrations" WHERE finished_at IS NOT NULL AND rolled_back_at IS NULL',
            )
        )
        existing: Final = connection.execute(
            "SELECT spec, migration_hash, generator_hash FROM litellm_perf_seed.checkpoint"
        ).fetchone()
        if existing is not None:
            assert SeedSpec.model_validate(existing[0]) == spec, "Resume requires the same seed specification"
            assert existing[1] == history, "Baseline migrations changed during seeding"
            assert existing[2] == generator_hash(), "Generator changed during seeding"
            return
        assert scalar(connection, b'SELECT EXISTS(SELECT 1 FROM "LiteLLM_SpendLogs" LIMIT 1)') is False, (
            "Refusing to seed an existing spend-log dataset"
        )
        connection.execute(
            "INSERT INTO litellm_perf_seed.checkpoint (spec, migration_hash, generator_hash) VALUES (%s::jsonb, %s, %s)",
            (spec.model_dump_json(), history, generator_hash()),
        )
        connection.execute(
            'INSERT INTO "LiteLLM_TeamTable" (team_id, team_alias, admins, members, models, updated_at) '
            "SELECT 'perf-team-' || g, 'perf-team-' || g, '{}', '{}', '{}', now() "
            "FROM generate_series(0, 99) AS g"
        )
        connection.execute(
            'INSERT INTO "LiteLLM_UserTable" (user_id, user_email, models, user_role) '
            "SELECT 'perf-user-' || g, 'perf-user-' || g || '@example.invalid', '{}', 'internal_user' "
            "FROM generate_series(0, 999) AS g"
        )
        connection.execute(
            'INSERT INTO "LiteLLM_EndUserTable" (user_id, models, blocked) '
            "SELECT 'perf-end-user-' || g, '{}', true FROM generate_series(0, 99999) AS g"
        )
        connection.execute(
            'INSERT INTO "LiteLLM_VerificationToken" (token, key_alias, user_id, team_id, models, blocked) '
            "SELECT md5('perf-key:' || g) || md5('perf-key-tail:' || g), 'perf-key-' || g, "
            "'perf-user-' || g / 10, 'perf-team-' || g / 100, '{}', true FROM generate_series(0, 9999) AS g"
        )


def seed_batch(connection: psycopg.Connection[tuple[object, ...]], spec: SeedSpec, rows: int) -> int:
    with connection.transaction():
        first: Final = INTEGER.validate_python(scalar(connection, b"SELECT rows + 1 FROM litellm_perf_seed.checkpoint"))
        last: Final = first + rows - 1
        connection.execute(
            'CREATE TEMP TABLE perf_seed_batch (LIKE "LiteLLM_SpendLogs" INCLUDING DEFAULTS) ON COMMIT DROP'
        )
        connection.execute(
            Path(__file__).with_suffix(".sql").read_bytes(), {"first": first, "last": last, "anchor": spec.anchor}
        )
        connection.execute('INSERT INTO "LiteLLM_SpendLogs" SELECT * FROM perf_seed_batch')
        for table, entity in (
            ("LiteLLM_DailyUserSpend", "user_id"),
            ("LiteLLM_DailyTeamSpend", "team_id"),
            ("LiteLLM_DailyEndUserSpend", "end_user_id"),
            ("LiteLLM_DailyGlobalSpend", ""),
            ("LiteLLM_DailyTagSpend", "tag"),
        ):
            aggregate_batch(connection, table, entity)
        connection.execute("UPDATE litellm_perf_seed.checkpoint SET rows = %s", (last,))
    return last


def aggregate_batch(connection: psycopg.Connection[tuple[object, ...]], table: str, entity: str) -> None:
    dimensions: Final = (
        *((entity,) if entity else ()),
        "date",
        *(("api_key",) if entity else ()),
        "model",
        "model_group",
        "custom_llm_provider",
        "mcp_namespaced_tool_name",
        "endpoint",
    )
    fields: Final = (
        "prompt_tokens",
        "completion_tokens",
        "spend",
        "api_requests",
        "successful_requests",
        "failed_requests",
        "total_response_time_ms",
        "timed_requests",
    )
    groups: Final = sql.SQL(", ").join(map(sql.Identifier, dimensions))
    totals: Final = sql.SQL(", ").join(map(sql.Identifier, fields))
    increments: Final = sql.SQL(", ").join(
        sql.SQL("{field} = {table}.{field} + EXCLUDED.{field}").format(
            field=sql.Identifier(field), table=sql.Identifier(table)
        )
        for field in fields
    )
    source: Final = sql.SQL(
        'SELECT *, "user" AS user_id, end_user AS end_user_id, to_char("startTime", \'YYYY-MM-DD\') AS date, '
        "'/v1/chat/completions'::text AS endpoint FROM perf_seed_batch"
    )
    connection.execute(
        sql.SQL(
            "WITH batch AS ({source}), grouped AS (SELECT {groups}, "
            "sum(prompt_tokens) AS prompt_tokens, sum(completion_tokens) AS completion_tokens, sum(spend) AS spend, "
            "count(*) AS api_requests, count(*) FILTER (WHERE status = 'success') AS successful_requests, "
            "count(*) FILTER (WHERE status = 'failure') AS failed_requests, "
            'sum(extract(epoch FROM ("endTime" - "startTime")) * 1000)::bigint AS total_response_time_ms, '
            "count(*) AS timed_requests FROM batch {tags} GROUP BY {groups}) "
            "INSERT INTO {table} (id, {groups}, {totals}, updated_at) "
            "SELECT 'perf-' || md5(jsonb_build_array({groups})::text), {groups}, {totals}, now() FROM grouped "
            "ON CONFLICT (id) DO UPDATE SET {increments}, updated_at = EXCLUDED.updated_at"
        ).format(
            source=source,
            groups=groups,
            totals=totals,
            table=sql.Identifier(table),
            increments=increments,
            tags=sql.SQL("CROSS JOIN LATERAL jsonb_array_elements_text(request_tags) AS tag")
            if entity == "tag"
            else sql.SQL(""),
        )
    )


def report_progress(progress: SeedProgress) -> None:
    sys.stdout.write(progress.model_dump_json() + "\n")
    sys.stdout.flush()


def checkpoint_progress(connection: psycopg.Connection[tuple[object, ...]]) -> SeedProgress:
    rows: Final = INTEGER.validate_python(scalar(connection, b"SELECT rows FROM litellm_perf_seed.checkpoint"))
    return measure(connection, rows)


def seed(
    connection: psycopg.Connection[tuple[object, ...]],
    spec: SeedSpec,
    batch_rows: int = BATCH_ROWS,
    report: Callable[[SeedProgress], None] = report_progress,
) -> SeedProgress:
    assert connection.autocommit, "Seeding needs an autocommit connection for bounded batch transactions"
    assert 0 < batch_rows <= 1_000_000, "Batch size must be between 1 and 1,000,000 rows"
    assert scalar(connection, b"SELECT pg_try_advisory_lock(%s)", (SEED_LOCK,)) is True, "Another seed is running"
    try:
        connection.execute("SET timezone = 'UTC'")
        connection.execute("SET search_path = public")
        initialize(connection, spec)
        while (progress := checkpoint_progress(connection)).total_bytes < spec.target_bytes or progress.rows == 0:
            report(progress)
            seed_batch(connection, spec, batch_rows)
        report(progress)
        connection.execute('VACUUM (ANALYZE) "LiteLLM_SpendLogs"')
        for table in (
            "LiteLLM_DailyUserSpend",
            "LiteLLM_DailyTeamSpend",
            "LiteLLM_DailyEndUserSpend",
            "LiteLLM_DailyGlobalSpend",
            "LiteLLM_DailyTagSpend",
        ):
            connection.execute(sql.SQL("VACUUM (ANALYZE) {}").format(sql.Identifier(table)))
        save_manifest(connection, spec)
        return measure(connection, progress.rows)
    finally:
        connection.execute("SELECT pg_advisory_unlock(%s)", (SEED_LOCK,))


def main() -> None:
    parser: Final = argparse.ArgumentParser(
        description="Seed a disposable, migrated PostgreSQL database without bodies"
    )
    parser.add_argument("--baseline-version", required=True)
    parser.add_argument("--target-bytes", type=int, default=TARGET_BYTES)
    parser.add_argument("--batch-rows", type=int, default=BATCH_ROWS)
    parser.add_argument("--anchor", required=True, help="Fixed UTC end of the 90-day window, e.g. 2026-10-15T00:00:00Z")
    parser.add_argument("--confirm-disposable", action="store_true", required=True)
    parser.add_argument("--manifest", type=Path)
    args: Final = SeedArguments.model_validate(vars(parser.parse_args()))
    spec: Final = SeedSpec.model_validate(args.model_dump(include=set(SeedSpec.model_fields)))
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        result: Final = seed(connection, spec, args.batch_rows)
        if args.manifest is not None:
            args.manifest.write_text(save_manifest(connection, spec).model_dump_json(indent=2) + "\n")
        sys.stdout.write(json.dumps({"spec": spec.model_dump(mode="json"), "result": result.model_dump()}) + "\n")


if __name__ == "__main__":
    main()
