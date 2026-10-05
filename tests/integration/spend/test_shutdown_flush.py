import json
import os
import signal
import threading
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import psycopg
import pytest
import yaml
from integration._support.client import Gateway, delete_key_if_present, eventually, string_value
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from psycopg import sql

REQUESTS_WHILE_BLOCKED: Final = 6
CANCEL_LOG_LINE: Final = "in-flight scheduled job(s) for shutdown"
BATCH_DRAINED_LOG_LINE: Final = f"flushed {REQUESTS_WHILE_BLOCKED} daily spend update items from in-memory queue"
MODEL_INSERT_ARRIVED_LOG_LINE: Final = "path=/model/new"
COMMIT_DELAY_SECONDS: Final = 15
BURST_REQUESTS: Final = 30


def _api_requests(table: str, column: str, identity: str) -> int:
    rows: Final = read_rows(
        f'SELECT coalesce(sum(api_requests), 0)::int AS total FROM "{table}" WHERE {column}=%s', (identity,)
    )
    total: Final = rows[0]["total"]
    assert isinstance(total, int)
    return total


def _waiting_on(table: str) -> int:
    rows: Final = read_rows(
        "SELECT count(*)::int AS waiting FROM pg_stat_activity WHERE wait_event_type='Lock' AND query LIKE %s",
        (f'%"{table}"%',),
    )
    waiting: Final = rows[0]["waiting"]
    assert isinstance(waiting, int)
    return waiting


def _committing_daily_user_spend() -> int:
    rows: Final = read_rows(
        "SELECT count(*)::int AS committing FROM pg_stat_activity "
        "WHERE query='COMMIT' AND state='active' AND wait_event='PgSleep' AND pid IN "
        "(SELECT pid FROM pg_locks WHERE relation = %s::regclass AND mode='RowExclusiveLock')",
        ('"LiteLLM_DailyUserSpend"',),
    )
    committing: Final = rows[0]["committing"]
    assert isinstance(committing, int)
    return committing


def _install_slow_commit(user_id: str, fails_once: bool) -> str:
    suffix: Final = f"slow_commit_{uuid.uuid4().hex}"
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SEQUENCE {}").format(sql.Identifier(suffix)))
        connection.execute(
            sql.SQL(
                "CREATE FUNCTION {}() RETURNS trigger LANGUAGE plpgsql AS $slow$ "
                "BEGIN PERFORM pg_sleep({}); "
                "IF {} AND nextval({}) = 1 THEN RAISE EXCEPTION 'integration: first COMMIT fails'; END IF; "
                "RETURN NULL; END $slow$"
            ).format(
                sql.Identifier(suffix), sql.Literal(COMMIT_DELAY_SECONDS), sql.Literal(fails_once), sql.Literal(suffix)
            )
        )
        connection.execute(
            sql.SQL(
                'CREATE CONSTRAINT TRIGGER {} AFTER INSERT OR UPDATE ON "LiteLLM_DailyUserSpend" '
                "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW WHEN (NEW.user_id = {}) EXECUTE FUNCTION {}()"
            ).format(sql.Identifier(suffix), sql.Literal(user_id), sql.Identifier(suffix))
        )
    return suffix


def _drop_slow_commit(suffix: str) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute(
            sql.SQL('DROP TRIGGER IF EXISTS {} ON "LiteLLM_DailyUserSpend"').format(sql.Identifier(suffix))
        )
        connection.execute(sql.SQL("DROP FUNCTION IF EXISTS {}()").format(sql.Identifier(suffix)))
        connection.execute(sql.SQL("DROP SEQUENCE IF EXISTS {}").format(sql.Identifier(suffix)))


def _provider(request: Request) -> Reply:
    if request.method != "POST":
        return Reply(status=404, body=b'{"error":"not scripted"}')
    assert request.target == "/v1/chat/completions"
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        ).encode()
    )


@dataclass(frozen=True, slots=True)
class _Shutdown:
    owner: str
    team: str
    owned: OwnedProxy
    key: str
    model: str

    def chat(self) -> None:
        body: Final = {"model": self.model, "messages": [{"role": "user", "content": f"spend {uuid.uuid4().hex}"}]}
        assert self.owned.gateway.request("POST", "/v1/chat/completions", body, key=self.key).status_code == 200

    def daily_user_requests(self) -> int:
        return _api_requests("LiteLLM_DailyUserSpend", "user_id", self.owner)

    def spend_logs(self) -> int:
        rows: Final = read_rows('SELECT count(*)::int AS total FROM "LiteLLM_SpendLogs" WHERE "user"=%s', (self.owner,))
        total: Final = rows[0]["total"]
        assert isinstance(total, int)
        return total

    def burst(self, requests: int) -> None:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for outcome in pool.map(lambda _: self.chat(), range(requests)):
                assert outcome is None

    def logged(self, line: str, times: int = 1) -> bool:
        return self.owned.log.read_text(errors="replace").count(line) >= times

    def chat_while_spend_update_is_blocked(self, blocker: psycopg.Connection, table: str) -> None:
        blocker.execute(f'LOCK TABLE "{table}" IN EXCLUSIVE MODE')
        for _ in range(REQUESTS_WHILE_BLOCKED):
            self.chat()
        eventually(lambda: _waiting_on(table), lambda waiting: waiting == 1, seconds=30)

    def start_blocked_model_insert(self) -> threading.Thread:
        body: Final = {
            "model_name": f"integration-blocked-{uuid.uuid4().hex}",
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "integration-provider-key"},
            "model_info": {},
        }

        def insert() -> None:
            try:
                self.owned.gateway.request("POST", "/model/new", body)
            except httpx.TransportError:
                pass

        thread: Final = threading.Thread(target=insert, daemon=True)
        thread.start()
        return thread

    def terminate_once(self, blocked: Callable[[], bool], release: Callable[[], None]) -> None:
        eventually(blocked, lambda state: state, seconds=60)
        self.owned.process.send_signal(signal.SIGTERM)
        eventually(lambda: self.logged(CANCEL_LOG_LINE), lambda seen: seen, seconds=60)
        release()
        self.owned.process.wait(timeout=120)


def _config_with_pool_limit(tmp_path: Path, pool_limit: int) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"]["database_connection_pool_limit"] = pool_limit
    config["general_settings"]["database_connection_pool_timeout"] = 60
    path: Final = tmp_path / f"pool-{pool_limit}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def _proxy_with_one_seeded_row(
    gateway: Gateway,
    tmp_path: Path,
    pool_limit: int,
    cancel_timeout_seconds: int = 5,
    settle_seconds: int = 0,
    requests: int = REQUESTS_WHILE_BLOCKED,
    workers: int = 1,
) -> Iterator[_Shutdown]:
    owner: Final = f"integration-owner-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario, wire_server(_provider) as wire:
        model: Final = scenario.model(api_base=wire.url + "/v1", num_retries=0)
        team: Final = scenario.team(models=[model])
        with owned_proxy_process(
            gateway,
            tmp_path,
            {
                "DATABASE_URL": os.environ["DATABASE_URL"],
                "LITELLM_LOG": "DEBUG",
                "GRACEFUL_SHUTDOWN_TIMEOUT": "1",
                "SCHEDULED_JOB_SHUTDOWN_FINISH_TIMEOUT_SECONDS": "1",
                "SCHEDULED_JOB_SHUTDOWN_CANCEL_TIMEOUT_SECONDS": str(cancel_timeout_seconds),
            },
            config=_config_with_pool_limit(tmp_path, pool_limit),
            remove_environment=("DATABASE_URL_READ_REPLICA",),
            workers=workers,
        ) as owned:
            key: Final = string_value(
                owned.gateway.post("/key/generate", {"user_id": owner, "team_id": team, "models": [model]})["key"]
            )
            scenario.cleanups.callback(delete_key_if_present, gateway, key)
            shutdown: Final = _Shutdown(owner, team, owned, key, model)
            shutdown.chat()
            eventually(shutdown.daily_user_requests, lambda total: total == 1, seconds=60)
            yield shutdown
    written: Final = 1 + requests
    if settle_seconds:
        eventually(
            lambda: (
                _api_requests("LiteLLM_DailyUserSpend", "user_id", owner),
                _api_requests("LiteLLM_DailyTeamSpend", "team_id", team),
            ),
            lambda totals: totals == (written, written),
            seconds=settle_seconds,
        )
    assert _api_requests("LiteLLM_DailyUserSpend", "user_id", owner) == written
    assert _api_requests("LiteLLM_DailyTeamSpend", "team_id", team) == written


@pytest.mark.covers("quota_management.spend_tracking.shutdown_cancel_keeps_in_flight_daily_batch")
def test_daily_spend_batch_cancelled_while_waiting_for_a_pool_connection_is_written_by_the_final_flush(
    gateway: Gateway, tmp_path: Path
) -> None:
    with (
        _proxy_with_one_seeded_row(gateway, tmp_path, pool_limit=2) as shutdown,
        psycopg.connect(os.environ["DATABASE_URL"]) as models,
        psycopg.connect(os.environ["DATABASE_URL"]) as memberships,
    ):
        models.execute('LOCK TABLE "LiteLLM_ProxyModelTable" IN EXCLUSIVE MODE')
        first: Final = shutdown.start_blocked_model_insert()
        eventually(lambda: _waiting_on("LiteLLM_ProxyModelTable"), lambda waiting: waiting == 1, seconds=30)
        shutdown.chat_while_spend_update_is_blocked(memberships, "LiteLLM_TeamMembership")
        second: Final = shutdown.start_blocked_model_insert()
        eventually(lambda: shutdown.logged(MODEL_INSERT_ARRIVED_LOG_LINE, times=2), lambda seen: seen, seconds=30)
        memberships.rollback()
        eventually(lambda: _waiting_on("LiteLLM_ProxyModelTable"), lambda waiting: waiting == 2, seconds=30)
        shutdown.terminate_once(lambda: shutdown.logged(BATCH_DRAINED_LOG_LINE), models.rollback)
        first.join(timeout=30)
        second.join(timeout=30)


@pytest.mark.covers("quota_management.spend_tracking.shutdown_cancel_keeps_in_flight_daily_batch")
def test_daily_spend_batch_cancelled_while_waiting_for_a_row_lock_is_written_exactly_once(
    gateway: Gateway, tmp_path: Path
) -> None:
    with (
        _proxy_with_one_seeded_row(gateway, tmp_path, pool_limit=10) as shutdown,
        psycopg.connect(os.environ["DATABASE_URL"]) as holder,
        psycopg.connect(os.environ["DATABASE_URL"]) as memberships,
    ):
        holder.execute('SELECT 1 FROM "LiteLLM_DailyUserSpend" WHERE user_id=%s FOR UPDATE', (shutdown.owner,))
        shutdown.chat_while_spend_update_is_blocked(memberships, "LiteLLM_TeamMembership")
        memberships.rollback()
        shutdown.terminate_once(
            lambda: shutdown.logged(BATCH_DRAINED_LOG_LINE) and _waiting_on("LiteLLM_DailyUserSpend") == 1,
            holder.rollback,
        )


@pytest.mark.parametrize(
    ("cancel_timeout_seconds", "commit_fails_once"),
    [
        pytest.param(60, False, id="cancel_budget_outlives_commit"),
        pytest.param(5, False, id="commit_outlives_cancel_budget"),
        pytest.param(60, True, id="commit_fails_within_cancel_budget"),
        pytest.param(5, True, id="commit_fails_after_cancel_budget"),
    ],
)
def test_daily_spend_batch_cancelled_while_postgres_is_committing_it_is_written_exactly_once(
    gateway: Gateway, tmp_path: Path, cancel_timeout_seconds: int, commit_fails_once: bool
) -> None:
    with (
        _proxy_with_one_seeded_row(
            gateway, tmp_path, pool_limit=10, cancel_timeout_seconds=cancel_timeout_seconds, settle_seconds=90
        ) as shutdown,
        psycopg.connect(os.environ["DATABASE_URL"]) as memberships,
    ):
        suffix: Final = _install_slow_commit(shutdown.owner, fails_once=commit_fails_once)
        try:
            shutdown.chat_while_spend_update_is_blocked(memberships, "LiteLLM_TeamMembership")
            memberships.rollback()
            shutdown.terminate_once(
                lambda: shutdown.logged(BATCH_DRAINED_LOG_LINE) and _committing_daily_user_spend() == 1,
                lambda: None,
            )
        finally:
            _drop_slow_commit(suffix)


def test_daily_spend_burst_across_two_workers_survives_shutdown_during_commit_exactly_once(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _proxy_with_one_seeded_row(
        gateway, tmp_path, pool_limit=10, settle_seconds=120, requests=BURST_REQUESTS, workers=2
    ) as shutdown:
        suffix: Final = _install_slow_commit(shutdown.owner, fails_once=False)
        try:
            shutdown.burst(BURST_REQUESTS)
            eventually(shutdown.spend_logs, lambda total: total == 1 + BURST_REQUESTS, seconds=60)
            shutdown.terminate_once(lambda: _committing_daily_user_spend() >= 1, lambda: None)
        finally:
            _drop_slow_commit(suffix)
