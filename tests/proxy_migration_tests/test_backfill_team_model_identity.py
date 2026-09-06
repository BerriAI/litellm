import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from scripts import backfill_team_model_identity as backfill

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/backfill_team_model_identity.py"
MIGRATIONS = ROOT / "litellm-proxy-extras/litellm_proxy_extras/migrations"
MIGRATION_NAMES = ("20260906000000_add_model_team_identity", "20260906000001_index_model_team_identity")
pytestmark = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="requires isolated PostgreSQL")


@dataclass
class Database:
    dsn: str
    schema: str
    conn: psycopg.Connection

    @property
    def models(self):
        return sql.Identifier(self.schema, "LiteLLM_ProxyModelTable")

    @property
    def teams(self):
        return sql.Identifier(self.schema, "LiteLLM_TeamTable")

    def seed(self, model_id="001", raw='{"team_id":"team-a"}', team_id=None, name="public-model"):
        self.conn.execute(
            sql.SQL(
                'INSERT INTO {} ("model_id", "model_name", "model_info", "team_id") VALUES (%s,%s,%s::jsonb,%s)'
            ).format(self.models),
            (model_id, name, raw, team_id),
        )

    def rows(self):
        return self.conn.execute(
            sql.SQL('SELECT "model_id", "team_id", to_jsonb(m) - \'team_id\' FROM {} m ORDER BY "model_id"').format(
                self.models
            )
        ).fetchall()

    def cli(self, *args, dsn=None):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--schema", self.schema, *args],
            env={**os.environ, "DATABASE_URL": dsn or self.dsn},
            text=True,
            capture_output=True,
            timeout=20,
        )
        assert result.stderr == "", result.stderr
        return result.returncode, tuple(json.loads(line) for line in result.stdout.splitlines())


@pytest.fixture
def db():
    schema = "team_identity_" + uuid.uuid4().hex
    dsn, _ = backfill.connection_parameters(os.environ["DATABASE_URL"], None)
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        conn.execute('CREATE TABLE "LiteLLM_TeamTable" ("team_id" TEXT PRIMARY KEY)')
        conn.execute(
            'CREATE TABLE "LiteLLM_ProxyModelTable" ("model_id" TEXT PRIMARY KEY, "model_name" TEXT NOT NULL, '
            '"model_info" JSONB, "litellm_params" JSONB NOT NULL DEFAULT \'{"api_key":"opaque-credential-sentinel"}\', '
            "\"updated_at\" TIMESTAMP NOT NULL DEFAULT '2020-01-01', \"created_at\" TIMESTAMP NOT NULL DEFAULT '2020-01-01', "
            '"created_by" TEXT NOT NULL DEFAULT \'operator\', "blocked" BOOLEAN NOT NULL DEFAULT false)'
        )
        conn.execute('ALTER TABLE "LiteLLM_ProxyModelTable" ADD COLUMN "updated_by" TEXT NOT NULL DEFAULT \'operator\'')
        for name in MIGRATION_NAMES:
            conn.execute((MIGRATIONS / name / "migration.sql").read_text())
        conn.execute("INSERT INTO \"LiteLLM_TeamTable\" VALUES ('team-a'), ('team-b')")
        try:
            yield Database(dsn, schema, conn)
        finally:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_default_dry_run_shows_exact_model_changes_and_writes_nothing(db):
    db.seed()
    before = db.rows()
    code, events = db.cli()
    assert code == 0
    assert db.rows() == before
    row = next(event for event in events if event["event"] == "model")
    assert row == {
        "event": "model",
        "model_id": "001",
        "model_name": "public-model",
        "status": "pending",
        "from_team_id": None,
        "to_team_id": "team-a",
        "action": "set_team_id",
    }
    assert events[-1]["pending"] == 1
    assert events[-1]["updated"] == 0
    assert "opaque-credential-sentinel" not in json.dumps(events)


def test_execute_is_idempotent_and_preserves_every_other_column(db):
    db.seed()
    db.seed("002", raw=None, name="global")
    before = db.rows()
    code, events = db.cli("--execute")
    assert code == 0
    assert [r[1] for r in db.rows()] == ["team-a", None]
    assert [r[2] for r in db.rows()] == [r[2] for r in before]
    assert events[-1]["updated"] == 1
    first = db.rows()
    code, events = db.cli("--execute")
    assert code == 0
    assert events[-1]["updated"] == 0
    assert db.rows() == first
    assert db.cli()[1][-1]["pending"] == 0


@pytest.mark.parametrize("raw", [None, "null", "{}", '{"team_id":null}'])
def test_global_models_never_acquire_a_team_from_their_names(db, raw):
    db.seed(raw=raw, name="model_name_team-a_fake-uuid")
    before = db.rows()
    assert db.cli("--execute")[0] == 0
    assert db.rows() == before


@pytest.mark.parametrize(
    "raw,existing,status",
    [
        ('{"team_id":"does-not-exist"}', None, "missing_team"),
        ('{"team_id":"does-not-exist"}', "does-not-exist", "missing_team"),
        ('{"team_id":123}', None, "invalid_team_id"),
        ('{"team_id":""}', None, "invalid_team_id"),
        ('{"team_public_model_name":"public"}', None, "invalid_team_id"),
        ("[]", None, "invalid_metadata"),
        ('"not-json"', None, "invalid_metadata"),
        ('{"team_id":"team-a"}', "team-b", "conflict"),
        ("{}", "team-a", "conflict"),
        (None, "team-a", "conflict"),
    ],
)
def test_ambiguous_rows_are_skipped_valid_rows_commit_and_exit_is_nonzero(db, raw, existing, status):
    db.seed("001", raw, existing)
    db.seed("002")
    before = db.rows()[0]
    code, events = db.cli("--execute")
    assert code == 2
    assert db.rows()[0] == before
    assert db.rows()[1][1] == "team-a"
    first = next(event for event in events if event.get("model_id") == "001")
    assert first["status"] == status
    assert first["action"] == "none"
    assert events[-1]["issues"] == 1
    assert db.cli()[0] == 2


def test_legacy_double_encoded_metadata_and_json_column_are_supported(db):
    db.conn.execute(
        sql.SQL('ALTER TABLE {} ALTER COLUMN "model_info" TYPE JSON USING "model_info"::json').format(db.models)
    )
    db.seed(raw=json.dumps(json.dumps({"team_id": "team-a", "unrelated": "retained"})))
    before = db.rows()[0][2]
    assert db.cli("--execute")[0] == 0
    assert db.rows()[0][1] == "team-a"
    assert db.rows()[0][2] == before


def test_unicode_quotes_whitespace_and_sql_like_identifiers_are_literal(db):
    owner = " 团队'; DROP TABLE anything; -- "
    db.conn.execute(sql.SQL("INSERT INTO {} VALUES (%s)").format(db.teams), (owner,))
    db.seed(model_id="id'\n\"", name='model\n"name', raw=json.dumps({"team_id": owner}))
    code, events = db.cli("--execute")
    assert code == 0
    assert db.rows()[0][1] == owner
    assert any(event.get("to_team_id") == owner for event in events)


@pytest.mark.parametrize("size", [1, 2, 3, 100])
def test_batches_visit_each_row_once_and_preserve_duplicate_public_names(db, size):
    for i in range(7):
        db.seed(f"{i:03}", raw=json.dumps({"team_id": "team-a" if i % 2 else "team-b"}))
    code, events = db.cli("--execute", "--batch-size", str(size))
    assert code == 0
    assert events[-1]["scanned"] == events[-1]["updated"] == 7
    assert events[-1]["batches"] == (7 + size - 1) // size
    assert len({event["model_id"] for event in events if event["event"] == "model"}) == 7


def test_bounded_run_can_resume_and_full_rerun_finds_new_earlier_ids(db):
    for i in range(4):
        db.seed(f"{i:03}")
    code, events = db.cli("--execute", "--batch-size", "2", "--max-batches", "1")
    assert code == 3
    assert events[-1]["scan_finished"] is False
    assert events[-1]["cursor"] == "001"
    db.seed("000-new")
    code, events = db.cli("--execute", "--after-model-id", "001")
    assert code == 0
    assert events[-1]["resumed"] is True
    assert events[-1]["updated"] == 2
    assert db.cli()[1][-1]["pending"] == 1
    assert db.cli("--execute")[1][-1]["updated"] == 1
    assert db.cli()[1][-1]["pending"] == 0


def test_empty_database_and_resume_past_end_are_noops(db):
    assert db.cli("--execute")[1][-1]["scanned"] == 0
    db.seed()
    code, events = db.cli("--execute", "--after-model-id", "zzz")
    assert code == 0
    assert events[-1]["scanned"] == 0
    assert db.rows()[0][1] is None


@pytest.mark.parametrize("change", ["metadata", "column", "delete", "team_delete"])
def test_concurrent_changes_are_not_overwritten(db, change):
    db.seed()
    with psycopg.connect(db.dsn, autocommit=True) as worker:
        rows = backfill.read_batch(worker, db.models, None, "zzz", 10)
        decisions = tuple(backfill.classify(row) for row in rows)
        if change == "metadata":
            db.conn.execute(
                sql.SQL('UPDATE {} SET "model_info"=%s::jsonb').format(db.models), ('{"team_id":"team-b"}',)
            )
        elif change == "column":
            db.conn.execute(sql.SQL("UPDATE {} SET \"team_id\"='team-b'").format(db.models))
        elif change == "delete":
            db.conn.execute(sql.SQL("DELETE FROM {}").format(db.models))
        else:
            db.conn.execute(sql.SQL("DELETE FROM {} WHERE \"team_id\"='team-a'").format(db.teams))
        before = db.rows()
        with worker.transaction():
            updated = backfill.execute_batch(worker, db.models, db.teams, decisions)
        assert updated == frozenset()
        assert db.rows() == before


def test_concurrent_rename_is_preserved(db):
    db.seed()
    rows = backfill.read_batch(db.conn, db.models, None, "zzz", 10)
    db.conn.execute(sql.SQL("UPDATE {} SET \"model_name\"='renamed'").format(db.models))
    assert backfill.execute_batch(
        db.conn, db.models, db.teams, tuple(backfill.classify(row) for row in rows)
    ) == frozenset({"001"})
    assert db.rows()[0][2]["model_name"] == "renamed"


def test_overlapping_workers_cannot_double_update(db):
    db.seed()
    with psycopg.connect(db.dsn, autocommit=True) as second:
        rows = backfill.read_batch(second, db.models, None, "zzz", 10)
        decisions = tuple(backfill.classify(row) for row in rows)
        assert db.cli("--execute")[0] == 0
        assert backfill.execute_batch(second, db.models, db.teams, decisions) == frozenset()
    assert db.cli("--execute")[1][-1]["updated"] == 0


def test_old_writer_ownership_change_is_reported_until_reconciled(db):
    db.seed()
    assert db.cli("--execute")[0] == 0
    db.conn.execute(sql.SQL('UPDATE {} SET "model_info"=%s::jsonb').format(db.models), ('{"team_id":"team-b"}',))
    assert db.cli("--execute")[0] == 2
    assert db.rows()[0][1] == "team-a"


def test_read_only_connections_can_preview_but_cannot_execute(db):
    db.seed()
    dsn = make_conninfo(db.dsn, options="-c default_transaction_read_only=on")
    assert db.cli(dsn=dsn)[0] == 0
    code, events = db.cli("--execute", dsn=dsn)
    assert code == 1
    assert events[-1]["reason"] == "execute_requires_writable_primary"
    assert db.rows()[0][1] is None


@pytest.mark.skipif(
    not os.environ.get("BACKFILL_TEST_REPLICA_URL"), reason="requires a physical standby of DATABASE_URL"
)
def test_physical_replica_can_preview_and_rejects_execution(db):
    db.seed()
    lsn = db.conn.execute("SELECT pg_current_wal_insert_lsn()::text").fetchone()[0]
    replica_dsn = os.environ["BACKFILL_TEST_REPLICA_URL"]
    with psycopg.connect(replica_dsn, autocommit=True) as replica:
        assert replica.execute("SELECT pg_is_in_recovery()").fetchone() == (True,)
        deadline = time.monotonic() + 10
        while not replica.execute("SELECT pg_last_wal_replay_lsn() >= %s::pg_lsn", (lsn,)).fetchone()[0]:
            assert time.monotonic() < deadline, "replica failed to catch up"
            time.sleep(0.02)
    code, events = db.cli(dsn=replica_dsn)
    assert code == 0
    assert events[0]["replica"] is True
    assert events[-1]["pending"] == 1
    code, events = db.cli("--execute", dsn=replica_dsn)
    assert code == 1
    assert events[-1]["reason"] == "execute_requires_writable_primary"
    assert db.rows()[0][1] is None


def test_duplicate_keys_in_legacy_json_are_reported_not_guessed(db):
    db.seed(raw=json.dumps('{"team_id":"team-a","team_id":"team-b"}'))
    code, events = db.cli("--execute")
    assert code == 2
    assert events[1]["status"] == "invalid_metadata"
    assert db.rows()[0][1] is None


@pytest.mark.parametrize("raw", ['{"team_id":"bad\\ud800id"}', '{"team_id":"bad\\u0000id"}'])
def test_unencodable_json_owner_is_skipped_while_valid_rows_commit(db, raw):
    db.conn.execute(
        sql.SQL('ALTER TABLE {} ALTER COLUMN "model_info" TYPE JSON USING "model_info"::json').format(db.models)
    )
    db.conn.execute(
        sql.SQL('INSERT INTO {} ("model_id", "model_name", "model_info") VALUES (\'001\', \'bad\', %s::json)').format(
            db.models
        ),
        (raw,),
    )
    db.seed("002")
    code, events = db.cli("--execute")
    assert code == 2
    assert events[1]["status"] == "invalid_team_id"
    assert db.conn.execute(sql.SQL('SELECT "team_id" FROM {} ORDER BY "model_id"').format(db.models)).fetchall() == [
        (None,),
        ("team-a",),
    ]


def test_large_scan_has_bounded_commits_and_no_missed_rows(db):
    db.conn.execute(
        sql.SQL(
            'INSERT INTO {} ("model_id", "model_name", "model_info") '
            "SELECT lpad(n::text, 6, '0'), 'shared-name', jsonb_build_object('team_id', 'team-a') "
            "FROM generate_series(1, 10001) n"
        ).format(db.models)
    )
    code, events = db.cli("--execute", "--batch-size", "10000")
    assert code == 0
    assert events[-1]["updated"] == events[-1]["scanned"] == 10001
    assert events[-1]["batches"] == 2
    assert all(row[1] == "team-a" for row in db.rows())


def test_schema_name_is_quoted(db):
    renamed = db.schema + ' "; SQL --'
    db.conn.execute(sql.SQL("ALTER SCHEMA {} RENAME TO {}").format(sql.Identifier(db.schema), sql.Identifier(renamed)))
    try:
        alternate = Database(db.dsn, renamed, db.conn)
        alternate.seed()
        assert alternate.cli("--execute")[0] == 0
        assert alternate.rows()[0][1] == "team-a"
    finally:
        db.conn.execute(
            sql.SQL("ALTER SCHEMA {} RENAME TO {}").format(sql.Identifier(renamed), sql.Identifier(db.schema))
        )


@pytest.mark.parametrize(
    "args", [("--execute", "--dry-run"), ("--batch-size", "0"), ("--max-batches", "0"), ("--lock-timeout-ms", "0")]
)
def test_invalid_cli_arguments_never_write(db, args):
    db.seed()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        env={**os.environ, "DATABASE_URL": db.dsn},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert db.rows()[0][1] is None


def test_actual_dry_run_transaction_is_read_only(db):
    db.seed()
    with psycopg.connect(db.dsn, autocommit=True) as worker:
        assert backfill.run(worker, backfill.Options(schema=db.schema), lambda event: None) == 0
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            worker.execute(sql.SQL("UPDATE {} SET \"team_id\"='team-a'").format(db.models))


@pytest.mark.parametrize("change", ["missing_column", "wrong_type", "nonunique_id", "missing_team_table", "view"])
def test_incompatible_schemas_fail_without_changing_model_data(db, change):
    db.seed()
    if change == "missing_column":
        db.conn.execute(sql.SQL('ALTER TABLE {} DROP COLUMN "team_id"').format(db.models))
    elif change == "wrong_type":
        db.conn.execute(sql.SQL('ALTER TABLE {} ALTER COLUMN "team_id" TYPE INTEGER USING NULL').format(db.models))
    elif change == "nonunique_id":
        db.conn.execute(sql.SQL('ALTER TABLE {} DROP CONSTRAINT "LiteLLM_ProxyModelTable_pkey"').format(db.models))
    elif change == "missing_team_table":
        db.conn.execute(sql.SQL("DROP TABLE {}").format(db.teams))
    else:
        db.conn.execute(sql.SQL('ALTER TABLE {} RENAME TO "source_models"').format(db.models))
        db.conn.execute(
            sql.SQL("CREATE VIEW {} AS SELECT * FROM {}").format(db.models, sql.Identifier(db.schema, "source_models"))
        )
    before = db.conn.execute(sql.SQL("SELECT to_jsonb(m) FROM {} m").format(db.models)).fetchall()
    assert db.cli("--execute")[0] == 1
    assert db.conn.execute(sql.SQL("SELECT to_jsonb(m) FROM {} m").format(db.models)).fetchall() == before


def test_lock_timeout_rolls_back_entire_batch_and_rerun_succeeds(db):
    db.seed("001")
    db.seed("002")
    with psycopg.connect(db.dsn) as blocker:
        blocker.execute(sql.SQL('UPDATE {} SET "model_name"="model_name" WHERE "model_id"=\'002\'').format(db.models))
        code, events = db.cli("--execute", "--lock-timeout-ms", "25")
        assert code == 1
        assert events[-1]["sqlstate"] == "55P03"
        assert all(row[1] is None for row in db.rows())
        blocker.rollback()
    assert db.cli("--execute")[1][-1]["updated"] == 2


def test_interruption_keeps_committed_batches_and_rolls_back_current_batch(db):
    db.seed("001")
    db.seed("002")
    with psycopg.connect(db.dsn) as blocker:
        blocker.execute(sql.SQL('UPDATE {} SET "model_name"="model_name" WHERE "model_id"=\'002\'').format(db.models))
        process = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT),
                "--schema",
                db.schema,
                "--execute",
                "--batch-size",
                "1",
                "--lock-timeout-ms",
                "10000",
            ],
            env={**os.environ, "DATABASE_URL": db.dsn},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                if json.loads(line)["event"] == "progress":
                    process.send_signal(signal.SIGINT)
                    break
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 130
            assert stderr == ""
            assert "interrupted_rerun_safely" in stdout
            assert [row[1] for row in db.rows()] == ["team-a", None]
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
            blocker.rollback()
    assert db.cli("--execute")[1][-1]["updated"] == 1


@pytest.mark.parametrize(
    "body,timeout,sqlstate",
    [
        ("PERFORM pg_sleep(0.1); RETURN NEW;", "10", "57014"),
        ("RAISE EXCEPTION 'SECRET-provider-credential';", "30000", "P0001"),
    ],
)
def test_statement_failure_is_atomic_and_server_details_are_redacted(db, body, timeout, sqlstate):
    db.seed()
    db.conn.execute(
        sql.SQL("CREATE FUNCTION {}() RETURNS trigger LANGUAGE plpgsql AS {}").format(
            sql.Identifier(db.schema, "test_before_update"), sql.Literal("BEGIN " + body + " END")
        )
    )
    db.conn.execute(
        sql.SQL("CREATE TRIGGER failure BEFORE UPDATE ON {} FOR EACH ROW EXECUTE FUNCTION {}()").format(
            db.models, sql.Identifier(db.schema, "test_before_update")
        )
    )
    code, events = db.cli("--execute", "--statement-timeout-ms", timeout)
    assert code == 1
    assert events[-1]["sqlstate"] == sqlstate
    assert "SECRET-provider-credential" not in json.dumps(events)
    assert db.rows()[0][1] is None


def test_schema_migrations_do_not_backfill_or_modify_existing_data(db):
    db.seed()
    before = db.rows()
    db.conn.execute(sql.SQL('ALTER TABLE {} DROP COLUMN "team_id"').format(db.models))
    for name in MIGRATION_NAMES:
        db.conn.execute((MIGRATIONS / name / "migration.sql").read_text())
    assert db.rows() == before
    valid = db.conn.execute(
        "SELECT indisvalid FROM pg_index WHERE indexrelid=to_regclass(%s)",
        (sql.Identifier(db.schema, "LiteLLM_ProxyModelTable_team_id_idx").as_string(db.conn),),
    ).fetchone()
    assert valid == (True,)


def test_no_update_permission_and_rls_cannot_silently_hide_rows(db):
    db.seed()
    role = "team_identity_role_" + uuid.uuid4().hex
    db.conn.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(role)))
    try:
        db.conn.execute(
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(db.schema), sql.Identifier(role))
        )
        db.conn.execute(
            sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(
                sql.Identifier(db.schema), sql.Identifier(role)
            )
        )
        dsn = make_conninfo(db.dsn, options=f"-c role={role}")
        assert db.cli(dsn=dsn)[0] == 0
        assert db.cli("--execute", dsn=dsn)[0] == 1
        assert db.rows()[0][1] is None
        db.conn.execute(sql.SQL('GRANT UPDATE ("team_id") ON {} TO {}').format(db.models, sql.Identifier(role)))
        assert db.cli("--execute", dsn=dsn)[0] == 0
        assert db.rows()[0][1] == "team-a"
        db.conn.execute(sql.SQL("ALTER TABLE {} ENABLE ROW LEVEL SECURITY").format(db.models))
        assert db.cli(dsn=dsn)[0] == 1
    finally:
        db.conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
        db.conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))


@pytest.mark.asyncio
async def test_actual_prisma_writes_keep_both_owners_in_sync(db):
    from prisma import Json, Prisma

    from litellm.repositories.model_repository import _TeamIdentityActions

    client = Prisma(datasource={"url": db.dsn + "?schema=" + db.schema})
    await client.connect()
    try:
        table = _TeamIdentityActions(client.litellm_proxymodeltable)
        data = {
            "model_id": "001",
            "model_name": "shared",
            "model_info": '{"team_id":"team-a"}',
            "litellm_params": "{}",
            "created_by": "operator",
            "updated_by": "operator",
        }
        row = await table.create(data)
        assert row.team_id == "team-a"
        assert row.model_info["team_id"] == "team-a"
        await table.update({"model_info": '{"team_id":"team-b"}'}, {"model_id": "001"})
        assert db.rows()[0][1] == "team-b"
        await table.update({"blocked": True}, {"model_id": "001"})
        assert db.rows()[0][1] == "team-b"
        assert (
            await table.create_many(({**data, "model_id": "002"}, {**data, "model_id": "003"}), skip_duplicates=True)
            == 2
        )
        assert (
            await table.update_many({"model_info": '{"team_id":"team-b"}'}, {"model_id": {"in": ["002", "003"]}}) == 2
        )
        assert all(row[1] == "team-b" for row in db.rows())
        await table.upsert({"model_id": "001"}, {"create": data, "update": {"model_info": "{}"}})
        assert db.rows()[0][1] is None
        await table.upsert({"model_id": "004"}, {"create": {**data, "model_id": "004"}, "update": {"blocked": True}})
        assert db.rows()[-1][1] == "team-a"
        await table.update({"model_info": Json({"team_id": "team-b"})}, {"model_id": "004"})
        assert db.rows()[-1][1] == "team-b"
        assert db.cli()[1][-1]["pending"] == 0
    finally:
        await client.disconnect()
