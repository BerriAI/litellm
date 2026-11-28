import hashlib
import os
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import psutil
import psycopg
import pytest
from psycopg.types.json import Jsonb
from pydantic import BaseModel, JsonValue, TypeAdapter
from redis import Redis

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.database import read_rows
from tests.integration._support.ports import allocate_port
from tests.integration._support.process import owned_proxy_process, private_state
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec, recording_service

MODEL: Final = "claude-opus-4-8"
MASTER_KEY: Final = "sk-affiliation-budget-parity"
ROOT: Final = Path(__file__).resolve().parents[4]


class Rates(BaseModel):
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None


@pytest.fixture(scope="module")
def actual_cost() -> float:
    prices: Final = TypeAdapter(dict[str, Rates]).validate_json(
        (ROOT / "model_prices_and_context_window.json").read_bytes()
    )
    rates: Final = prices[MODEL]
    assert rates.input_cost_per_token is not None and rates.output_cost_per_token is not None
    return 4 * rates.input_cost_per_token + 2 * rates.output_cost_per_token


@dataclass(frozen=True)
class Case:
    owner: bool = False
    default: bool = False
    maximum: float = 0.5
    spend: float = 0.0
    rollover: bool = False
    increase: bool = False
    status: int = 200


@dataclass(frozen=True)
class Identity:
    token: str
    counter: str
    user: str
    team: str | None

    def persisted(self, database: str) -> float:
        rows: Final = (
            read_rows('SELECT spend FROM "LiteLLM_UserTable" WHERE user_id = %s', (self.user,), database_url=database)
            if self.team is None
            else read_rows(
                'SELECT spend FROM "LiteLLM_TeamMembership" WHERE user_id = %s AND team_id = %s',
                (self.user, self.team),
                database_url=database,
            )
        )
        value: Final = rows[0]["spend"]
        assert isinstance(value, (int, float))
        return float(value)


def _identity(database: str, case: Case) -> Identity:
    identifier: Final = uuid.uuid4().hex
    token: Final = f"sk-{identifier}"
    hashed: Final = hashlib.sha256(token.encode()).hexdigest()
    team: Final = None if case.owner else f"team-{identifier}"
    duration: Final = "30d" if case.rollover else None
    ceiling: Final = 2.0 if case.rollover else None
    credit: Final = 1.0 if case.rollover else None
    with psycopg.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO "LiteLLM_UserTable"
                (user_id, user_role, spend, max_budget, budget_duration, rollover_max_budget, rollover_credit)
            VALUES (%s, 'internal_user', %s, %s, %s, %s, %s)
            """,
            (
                identifier,
                case.spend if case.owner else 0,
                case.maximum if case.owner else None,
                duration if case.owner else None,
                ceiling if case.owner else None,
                credit if case.owner else None,
            ),
        )
        if team is not None:
            budget: Final = f"budget-{identifier}"
            connection.execute(
                """
                INSERT INTO "LiteLLM_BudgetTable"
                    (budget_id, max_budget, created_by, updated_by, updated_at, budget_duration,
                     rollover_max_budget, temp_budget_increase, temp_budget_expiry)
                VALUES (%s, %s, 'parity', 'parity', NOW(), %s, %s, %s,
                        CASE WHEN %s THEN NOW() + INTERVAL '1 day' END)
                """,
                (budget, case.maximum, duration, ceiling, 1.0 if case.increase else None, case.increase),
            )
            connection.execute(
                """
                INSERT INTO "LiteLLM_TeamTable" (team_id, metadata, updated_at)
                VALUES (%s, jsonb_strip_nulls(jsonb_build_object('team_member_budget_id', %s::text)), NOW())
                """,
                (team, budget if case.default else None),
            )
            connection.execute(
                """
                INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, budget_id, spend, rollover_credit)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (identifier, team, None if case.default else budget, case.spend, credit),
            )
        connection.execute(
            'INSERT INTO "LiteLLM_VerificationToken" (token, user_id, team_id) VALUES (%s, %s, %s)',
            (hashed, identifier, team),
        )
    counter: Final = f"spend:team_member:{identifier}:{team}" if team is not None else f"spend:user:{identifier}"
    return Identity(token, counter, identifier, team)


@contextmanager
def _rust(binary: Path, config: Path, environment: dict[str, str]) -> Iterator[httpx.Client]:
    port: Final = allocate_port()
    with (
        config.with_suffix(".rust.log").open("w") as log,
        subprocess.Popen(
            [str(binary)],
            env={**os.environ, **environment, "HOST": "127.0.0.1", "PORT": str(port), "LITELLM_CONFIG": str(config)},
            stdout=log,
            stderr=subprocess.STDOUT,
        ) as process,
        httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=15) as client,
    ):

        def ready() -> bool:
            assert process.poll() is None, config.with_suffix(".rust.log").read_text()
            try:
                client.get("/health/readiness")
            except httpx.ConnectError:
                return False
            return True

        try:
            eventually(ready, bool, seconds=30)
            yield client
        finally:
            process.terminate()
            process.wait(timeout=15)


@dataclass(frozen=True)
class Proxies:
    python: httpx.Client
    python_pid: int
    rust: httpx.Client
    database: str
    redis: Redis
    upstream: RecordingServer


@pytest.fixture(scope="module")
def proxies(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Proxies]:
    binary: Final = Path(
        os.environ.get("LITELLM_PARITY_GATEWAY_BINARY", ROOT / "litellm/rust/target/debug/litellm-gateway")
    )
    if not binary.is_file() or "DATABASE_URL" not in os.environ:
        pytest.skip("Live budget parity requires DATABASE_URL and a built litellm-gateway binary")
    directory: Final = tmp_path_factory.mktemp("affiliation-budget-parity")
    config: Final = directory / "proxy.yaml"
    with recording_service() as upstream, private_state(directory, os.environ) as state, httpx.Client() as unused:
        upstream.expected_requests = None
        config.write_text(
            f"""model_list:
  - model_name: metered
    litellm_params:
      model: anthropic/{MODEL}
      api_key: parity-mock
      api_base: {upstream.base_url}
general_settings:
  master_key: {MASTER_KEY}
  disable_budget_reservation: false
"""
        )
        environment: Final = {
            **state.environment,
            "LITELLM_RUST_FIRST": "0",
            "LITELLM_RUST": "0",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "RUST_LOG": "debug",
            "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT / "tests"), os.environ.get("PYTHONPATH")))),
            "STORE_MODEL_IN_DB": "False",
            "PROXY_BATCH_WRITE_AT": "1",
            "LITELLM_ACCOUNTING_DELIVERY_BATCH_SIZE": "1",
            "SPEND_ROLLUP_LOCK_TIMEOUT_MS": "20000",
        }
        gateway: Final = Gateway(unused, MASTER_KEY, upstream.base_url)
        with (
            owned_proxy_process(gateway, directory, environment, config=config) as python,
            _rust(binary, config, environment) as native,
            Redis(host=environment["REDIS_HOST"], port=int(environment["REDIS_PORT"])) as redis,
        ):
            yield Proxies(python.gateway.client, python.process.pid, native, state.database_url, redis, upstream)


def _outcome(client: httpx.Client, proxies: Proxies, case: Case, actual_cost: float) -> tuple[int, float, float]:
    identity: Final = _identity(proxies.database, case)
    proxies.upstream.default_response = ResponseSpec(
        {
            "id": f"msg_{identity.user}",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 4, "output_tokens": 2},
        }
    )
    proxies.redis.set(identity.counter, case.spend)
    response: Final = client.post(
        "/v1/messages",
        headers={"Authorization": f"Bearer {identity.token}", "anthropic-version": "2023-06-01"},
        json={"model": "metered", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 8},
    )
    assert response.status_code == case.status, response.text
    expected: Final = case.spend + (actual_cost if case.status == 200 else 0)

    def counter() -> float:
        value: Final = proxies.redis.get(identity.counter)
        assert isinstance(value, bytes)
        return float(value)

    settled: Final = eventually(counter, lambda value: value == pytest.approx(expected))
    persisted: Final = eventually(
        lambda: identity.persisted(proxies.database), lambda value: value == pytest.approx(expected), seconds=30
    )
    return response.status_code, settled, persisted


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(Case(owner=True), id="owner-under-cap"),
        pytest.param(Case(owner=True, spend=0.6, status=422), id="owner-over-cap"),
        pytest.param(Case(owner=True, spend=0.6, rollover=True), id="owner-rollover"),
        pytest.param(Case(), id="member-under-cap"),
        pytest.param(Case(spend=0.6, status=422), id="member-over-cap"),
        pytest.param(Case(default=True), id="member-default"),
        pytest.param(Case(default=True, spend=0.6, status=422), id="member-default-over-cap"),
        pytest.param(Case(default=True, maximum=0), id="member-zero-default"),
        pytest.param(Case(spend=0.6, rollover=True), id="member-rollover"),
        pytest.param(Case(spend=0.6, increase=True), id="member-temporary-increase"),
    ],
)
def test_affiliation_budget_admission_and_settlement_match_python(
    proxies: Proxies, case: Case, actual_cost: float
) -> None:
    python: Final = _outcome(proxies.python, proxies, case, actual_cost)
    native: Final = _outcome(proxies.rust, proxies, case, actual_cost)
    assert native == pytest.approx(python)


@pytest.mark.parametrize("case", [pytest.param(Case(owner=True), id="owner"), pytest.param(Case(), id="member")])
def test_standalone_applies_affiliation_spend_while_python_is_paused(
    proxies: Proxies, case: Case, actual_cost: float
) -> None:
    python: Final = _outcome(proxies.python, proxies, case, actual_cost)
    process: Final = psutil.Process(proxies.python_pid)
    process.suspend()
    try:
        eventually(process.status, lambda status: status == psutil.STATUS_STOPPED)
        native: Final = _outcome(proxies.rust, proxies, case, actual_cost)
        assert native == pytest.approx(python)
    finally:
        process.resume()


def test_python_and_rust_claim_concurrently_without_applying_a_delivery_twice(proxies: Proxies) -> None:
    identity: Final = _identity(proxies.database, Case(owner=True))
    first: Final = uuid.uuid4().hex
    second: Final = uuid.uuid4().hex
    with psycopg.connect(proxies.database) as locked, psycopg.connect(proxies.database) as enqueue:
        locked.execute('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s FOR UPDATE', (identity.user,))
        enqueue.execute(
            """
            INSERT INTO "LiteLLM_SpendLogDelivery" (delivery_id, request_id, is_batch, accounting_payload)
            VALUES (%s, %s, false, %s), (%s, %s, false, %s)
            """,
            (
                first,
                first,
                Jsonb({"spend": 0.125, "metadata": {"accounting_entities": {"user": identity.user}}}),
                second,
                second,
                Jsonb({"spend": 0.25, "metadata": {"accounting_entities": {"user": identity.user}}}),
            ),
        )
        enqueue.commit()

        def blocked_workers() -> set[str]:
            rows: Final = read_rows(
                """
                SELECT query FROM pg_stat_activity
                WHERE datname = current_database() AND state = 'active' AND wait_event_type = 'Lock'
                  AND strpos(query, 'UPDATE') > 0 AND strpos(query, 'LiteLLM_UserTable') > 0
                """,
                (),
                database_url=proxies.database,
            )
            return {"rust" if "jsonb_to_recordset" in str(row["query"]) else "python" for row in rows}

        assert eventually(blocked_workers, lambda workers: workers == {"python", "rust"}, seconds=15) == {
            "python",
            "rust",
        }

    def delivered() -> list[dict[str, JsonValue]]:
        return read_rows(
            """
            SELECT state::text, attempts FROM "LiteLLM_SpendLogDelivery"
            WHERE delivery_id IN (%s, %s) ORDER BY delivery_id
            """,
            (first, second),
            database_url=proxies.database,
        )

    expected: Final = [{"state": "applied", "attempts": 0}, {"state": "applied", "attempts": 0}]
    assert eventually(delivered, lambda rows: rows == expected) == expected
    assert identity.persisted(proxies.database) == pytest.approx(0.375)
