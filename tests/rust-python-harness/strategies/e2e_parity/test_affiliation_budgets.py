import hashlib
import json
import os
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import httpx
import psutil
import psycopg
import pytest
from psycopg.types.json import Jsonb
from pydantic import BaseModel, JsonValue, TypeAdapter
from redis import Redis

from litellm.proxy.common_utils.user_api_key_cache import END_USER_RESTRICTED_REGISTRY_OVERFLOW_SENTINEL
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
                (user_id, user_role, spend, max_budget, budget_duration, rollover_max_budget, rollover_credit,
                 budget_reset_at)
            VALUES (%s, 'internal_user', %s, %s, %s, %s, %s, CASE WHEN %s THEN NOW() + INTERVAL '1 day' END)
            """,
            (
                identifier,
                case.spend if case.owner else 0,
                case.maximum if case.owner else None,
                duration if case.owner else None,
                ceiling if case.owner else None,
                credit if case.owner else None,
                case.owner and case.rollover,
            ),
        )
        if team is not None:
            budget: Final = f"budget-{identifier}"
            connection.execute(
                """
                INSERT INTO "LiteLLM_BudgetTable"
                    (budget_id, max_budget, created_by, updated_by, updated_at, budget_duration,
                     rollover_max_budget, temp_budget_increase, temp_budget_expiry, budget_reset_at)
                VALUES (%s, %s, 'parity', 'parity', NOW(), %s, %s, %s,
                        CASE WHEN %s THEN NOW() + INTERVAL '1 day' END,
                        CASE WHEN %s THEN NOW() + INTERVAL '1 day' END)
                """,
                (budget, case.maximum, duration, ceiling, 1.0 if case.increase else None, case.increase, case.rollover),
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
def _rust(
    binary: Path, config: Path, environment: dict[str, str], remove_environment: tuple[str, ...] = ()
) -> Iterator[tuple[httpx.Client, int]]:
    port: Final = allocate_port()
    with (
        config.with_suffix(".rust.log").open("w") as log,
        subprocess.Popen(
            [str(binary)],
            env={
                **{name: value for name, value in os.environ.items() if name not in remove_environment},
                **environment,
                "HOST": "127.0.0.1",
                "PORT": str(port),
                "LITELLM_CONFIG": str(config),
            },
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
            yield client, process.pid
        finally:
            process.terminate()
            process.wait(timeout=15)


@dataclass(frozen=True)
class Proxies:
    python: httpx.Client
    python_pid: int
    rust: httpx.Client
    rust_pid: int
    database: str
    redis: Redis
    upstream: RecordingServer
    uses_redis: bool


@pytest.fixture(scope="module")
def proxies(tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest) -> Iterator[Proxies]:
    binary: Final = Path(
        os.environ.get("LITELLM_PARITY_GATEWAY_BINARY", ROOT / "litellm/rust/target/debug/litellm-gateway")
    )
    if not binary.is_file() or "DATABASE_URL" not in os.environ:
        pytest.skip("Live budget parity requires DATABASE_URL and a built litellm-gateway binary")
    directory: Final = tmp_path_factory.mktemp("affiliation-budget-parity")
    config: Final = directory / "proxy.yaml"
    uses_redis: Final = getattr(request, "param", True) is True
    remove_environment: Final = () if uses_redis else tuple(name for name in os.environ if name.startswith("REDIS_"))
    with recording_service() as upstream, private_state(directory, os.environ) as state, httpx.Client() as unused:
        upstream.expected_requests = None
        config.write_text(
            f"""model_list:
  - model_name: metered
    litellm_params:
      model: anthropic/{MODEL}
      api_key: parity-mock
      api_base: {upstream.base_url}
    model_info:
      access_groups: [metered-family]
  - model_name: budget-primary
    litellm_params:
      model: anthropic/{MODEL}
      api_key: parity-mock
      api_base: {upstream.base_url}
  - model_name: budget-secondary
    litellm_params:
      model: anthropic/{MODEL}
      api_key: parity-mock
      api_base: {upstream.base_url}
    model_info:
      id: budget-secondary-id
router_settings:
  model_group_alias:
    metered-alias: metered
    budget-secondary-alias: budget-secondary
  fallbacks:
    - budget-primary: [budget-secondary-alias]
general_settings:
  master_key: {MASTER_KEY}
  disable_budget_reservation: false
litellm_settings:
  enable_redis_auth_cache: {str(uses_redis).lower()}
  budget_rollover: true
"""
        )
        environment: Final = {
            **{name: value for name, value in state.environment.items() if uses_redis or not name.startswith("REDIS_")},
            "LITELLM_RUST_FIRST": "0",
            "LITELLM_RUST": "0",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "RUST_LOG": "debug",
            "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT / "tests"), os.environ.get("PYTHONPATH")))),
            "STORE_MODEL_IN_DB": "False",
            "PROXY_BATCH_WRITE_AT": "1",
            "PROXY_BUDGET_RESCHEDULER_MIN_TIME": "2",
            "PROXY_BUDGET_RESCHEDULER_MAX_TIME": "2",
            "LITELLM_ACCOUNTING_DELIVERY_BATCH_SIZE": "1",
            "SPEND_ROLLUP_LOCK_TIMEOUT_MS": "20000",
        }
        gateway: Final = Gateway(unused, MASTER_KEY, upstream.base_url)
        with (
            owned_proxy_process(
                gateway, directory, environment, config=config, remove_environment=remove_environment
            ) as python,
            _rust(binary, config, environment, remove_environment) as (native, native_pid),
            Redis(host=state.environment["REDIS_HOST"], port=int(state.environment["REDIS_PORT"])) as redis,
        ):
            # Cases insert end users directly, without the management write that invalidates the registry.
            redis.set("end_user_restricted_registry", json.dumps(END_USER_RESTRICTED_REGISTRY_OVERFLOW_SENTINEL))
            yield Proxies(
                python.gateway.client,
                python.process.pid,
                native,
                native_pid,
                state.database_url,
                redis,
                upstream,
                uses_redis,
            )


@contextmanager
def _paused_between_sweeps(pid: int, redis: Redis) -> Iterator[None]:
    process: Final = psutil.Process(pid)
    lease: Final = "cronjob_lock:reset_budget_job"
    eventually(lambda: redis.set(lease, '"parity-test"', nx=True, ex=30), bool)
    try:
        process.suspend()
        try:
            eventually(process.status, lambda status: status == psutil.STATUS_STOPPED)
            redis.delete(lease)
            yield
        finally:
            process.resume()
    finally:
        redis.eval(
            "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end",
            1,
            lease,
            '"parity-test"',
        )


def _window_outcome(
    proxies: Proxies, client: httpx.Client, ended: datetime, actual_cost: float, scope: str
) -> tuple[datetime, datetime, float, float, float]:
    identity: Final = _identity(proxies.database, Case(owner=scope != "team"))
    key_hash: Final = hashlib.sha256(identity.token.encode()).hexdigest()
    old_start: Final = ended - timedelta(hours=1)
    entity: Final = {"key": key_hash, "user": identity.user, "team": identity.team}[scope]
    assert entity is not None
    counter: Final = f"spend:user_window:{entity}:1h" if scope == "user" else f"spend:{scope}:{entity}:window:1h"
    windows: Final = Jsonb([{"budget_duration": "1h", "max_budget": 1, "reset_at": ended.isoformat()}])
    with psycopg.connect(proxies.database) as connection:
        statement: Final = {
            "key": 'UPDATE "LiteLLM_VerificationToken" SET budget_limits = %s WHERE token = %s',
            "user": 'UPDATE "LiteLLM_UserTable" SET budget_limits = %s WHERE user_id = %s',
            "team": 'UPDATE "LiteLLM_TeamTable" SET budget_limits = %s WHERE team_id = %s',
        }[scope]
        connection.execute(statement, (windows, entity))
        connection.execute(
            """INSERT INTO "LiteLLM_BudgetWindowSpend"
                   (entity_type, entity_id, window_duration, window_start, spend, created_at, updated_at)
               VALUES (%s, %s, '1h', %s, 1.5, NOW(), NOW())""",
            (scope, entity, old_start.replace(tzinfo=None)),
        )
        proxies.redis.set(counter, "1.5")

    def row() -> tuple[datetime, datetime, float, float]:
        rows: Final = read_rows(
            """WITH budgets AS (
                   SELECT user_id AS id, budget_limits, 'user' AS scope FROM "LiteLLM_UserTable"
                   UNION ALL SELECT team_id, budget_limits, 'team' FROM "LiteLLM_TeamTable"
                   UNION ALL SELECT token, budget_limits, 'key' FROM "LiteLLM_VerificationToken"
               ) SELECT b.budget_limits->0->>'reset_at' AS reset_at,
                      w.window_start::text AS window_start, w.spend
               FROM budgets b JOIN "LiteLLM_BudgetWindowSpend" w ON b.id = w.entity_id AND b.scope = w.entity_type
               WHERE b.id = %s AND b.scope = %s AND w.window_duration = '1h'""",
            (entity, scope),
            database_url=proxies.database,
        )
        reset_at, window_start, spend = (rows[0][key] for key in ("reset_at", "window_start", "spend"))
        raw: Final = proxies.redis.get(counter)
        assert isinstance(reset_at, str) and isinstance(window_start, str)
        assert isinstance(spend, (int, float)) and isinstance(raw, bytes)
        return datetime.fromisoformat(reset_at), datetime.fromisoformat(window_start), float(spend), float(raw)

    reset: Final = eventually(row, lambda row: row[0] > ended and row[3] == 0.5)
    assert reset == (ended + timedelta(hours=1), ended.replace(tzinfo=None), 0.0, 0.5)
    _provider_response(proxies, identity)
    response: Final = _send(client, identity)
    assert response.status_code == 200, response.text
    result: Final = eventually(row, lambda row: row[2:] == pytest.approx((actual_cost, 0.5 + actual_cost)), seconds=30)
    assert result[:2] == reset[:2]
    proxies.redis.set(counter, "1.5")
    denied: Final = _send(client, identity)
    assert denied.status_code == 422, denied.text
    persisted: Final = eventually(
        lambda: identity.persisted(proxies.database), lambda value: value == pytest.approx(actual_cost), seconds=30
    )
    return (*result, persisted)


@pytest.mark.parametrize("scope", ["user", "team", "key"])
def test_standalone_advances_window_and_redis_counter_while_python_is_paused(
    proxies: Proxies, actual_cost: float, scope: str
) -> None:
    ended: Final = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=2)
    with _paused_between_sweeps(proxies.rust_pid, proxies.redis):
        python: Final = _window_outcome(proxies, proxies.python, ended, actual_cost, scope)
    with _paused_between_sweeps(proxies.python_pid, proxies.redis):
        rust: Final = _window_outcome(proxies, proxies.rust, ended, actual_cost, scope)
    assert rust[:2] == python[:2]
    assert rust[2:] == pytest.approx(python[2:])


@pytest.mark.parametrize("proxies", [False], ids=["postgres"], indirect=True)
def test_key_window_refuses_using_maintained_spend_without_redis(proxies: Proxies) -> None:
    identity: Final = _identity(proxies.database, Case(owner=True))
    token_hash: Final = hashlib.sha256(identity.token.encode()).hexdigest()
    start: Final = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=2)
    reset: Final = start + timedelta(hours=1)
    windows: Final = Jsonb([{"budget_duration": "1h", "max_budget": 1, "reset_at": reset.isoformat()}])
    with psycopg.connect(proxies.database) as connection:
        connection.execute(
            'UPDATE "LiteLLM_VerificationToken" SET budget_limits = %s WHERE token = %s', (windows, token_hash)
        )
        connection.execute(
            """INSERT INTO "LiteLLM_BudgetWindowSpend"
                   (entity_type, entity_id, window_duration, window_start, spend, created_at, updated_at)
               VALUES ('key', %s, '1h', %s, 1.5, NOW(), NOW())""",
            (token_hash, start.replace(tzinfo=None)),
        )
    python: Final = _send(proxies.python, identity)
    rust: Final = _send(proxies.rust, identity)
    assert python.status_code == rust.status_code == 422, (python.text, rust.text)
    assert identity.persisted(proxies.database) == 0.0


def _provider_response(proxies: Proxies, identity: Identity) -> None:
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


def _send(
    client: httpx.Client, identity: Identity, end_user: str | None = None, model: str = "metered"
) -> httpx.Response:
    return client.post(
        "/v1/messages",
        headers={"Authorization": f"Bearer {identity.token}", "anthropic-version": "2023-06-01"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8,
            **({"metadata": {"user_id": end_user}} if end_user is not None else {}),
        },
    )


def _model_budget_identity(
    proxies: Proxies, maximum: float, scope: str, entry: str
) -> tuple[Identity, str | None, str]:
    identity: Final = _identity(
        proxies.database, Case(owner=scope in ("key", "owner", "end-user"), default=scope == "default", maximum=2.0)
    )
    hashed: Final = hashlib.sha256(identity.token.encode()).hexdigest()
    end_user: Final = f"customer-{identity.user}" if scope == "end-user" else None
    budget_id: Final = f"budget-{identity.user}"
    policy: Final = Jsonb({entry: {"budget_limit": maximum, "time_period": "1h"}})
    with psycopg.connect(proxies.database) as connection:
        if scope == "key":
            connection.execute(
                'UPDATE "LiteLLM_VerificationToken" SET model_max_budget = %s WHERE token = %s', (policy, hashed)
            )
        elif scope == "owner":
            connection.execute(
                'UPDATE "LiteLLM_UserTable" SET model_max_budget = %s WHERE user_id = %s', (policy, identity.user)
            )
        elif scope == "team":
            connection.execute(
                'UPDATE "LiteLLM_TeamTable" SET model_max_budget = %s WHERE team_id = %s', (policy, identity.team)
            )
        elif end_user is not None:
            connection.execute(
                """INSERT INTO "LiteLLM_BudgetTable" (budget_id, model_max_budget, created_by, updated_by, updated_at)
                   VALUES (%s, %s, 'parity', 'parity', NOW())""",
                (budget_id, policy),
            )
            connection.execute(
                'INSERT INTO "LiteLLM_EndUserTable" (user_id, budget_id) VALUES (%s, %s)', (end_user, budget_id)
            )
        else:
            connection.execute(
                'UPDATE "LiteLLM_BudgetTable" SET model_max_budget = %s WHERE budget_id = %s', (policy, budget_id)
            )
    prefix: Final = {
        "key": f"virtual_key_spend:{hashed}",
        "owner": f"user_model_spend:{identity.user}",
        "team": f"team_model_spend:{identity.team}",
        "end-user": f"end_user_model_spend:{end_user}",
    }.get(scope, f"team_member_model_spend:{len(identity.user)}:{identity.user}:{identity.team}")
    counter: Final = f"{prefix}:{entry}:1h"
    return identity, end_user, counter


def _affiliation_lifetime_outcome(
    proxies: Proxies, client: httpx.Client, scope: str, spend: float, expected_cost: float
) -> int:
    identity: Final = _identity(proxies.database, Case(owner=True))
    identifier: Final = identity.user
    hashed: Final = hashlib.sha256(identity.token.encode()).hexdigest()
    end_user: Final = identifier if scope == "end-user" else None
    counter: Final = f"spend:{'end_user' if scope == 'end-user' else scope}:{identifier}"
    with psycopg.connect(proxies.database) as connection:
        connection.execute(
            """INSERT INTO "LiteLLM_BudgetTable"
               (budget_id, max_budget, rollover_max_budget, temp_budget_increase, temp_budget_expiry,
                allowed_models, created_by, updated_by, updated_at)
               VALUES (%s, 1, 2, 2, NOW() + INTERVAL '1 day', ARRAY['budget-primary'], 'parity', 'parity', NOW())""",
            (identifier,),
        )
        if scope == "org":
            connection.execute(
                """INSERT INTO "LiteLLM_OrganizationTable"
                   (organization_id, organization_alias, budget_id, spend, created_by, updated_by)
                   VALUES (%s, %s, %s, %s, 'parity', 'parity')""",
                (identifier, identifier, identifier, spend),
            )
            connection.execute(
                'UPDATE "LiteLLM_VerificationToken" SET organization_id = %s WHERE token = %s', (identifier, hashed)
            )
        elif scope == "project":
            connection.execute(
                """INSERT INTO "LiteLLM_ProjectTable" (project_id, budget_id, spend, created_by, updated_by)
                   VALUES (%s, %s, %s, 'parity', 'parity')""",
                (identifier, identifier, spend),
            )
            connection.execute(
                'UPDATE "LiteLLM_VerificationToken" SET project_id = %s WHERE token = %s', (identifier, hashed)
            )
        else:
            connection.execute(
                'INSERT INTO "LiteLLM_EndUserTable" (user_id, budget_id, spend) VALUES (%s, %s, %s)',
                (identifier, identifier, spend),
            )
    proxies.redis.set(counter, spend)
    _provider_response(proxies, identity)
    response: Final = _send(client, identity, end_user)
    expected: Final = 200 if expected_cost else 422
    assert response.status_code == expected, response.text
    query: Final = {
        "org": 'SELECT spend FROM "LiteLLM_OrganizationTable" WHERE organization_id = %s',
        "project": 'SELECT spend FROM "LiteLLM_ProjectTable" WHERE project_id = %s',
        "end-user": 'SELECT spend FROM "LiteLLM_EndUserTable" WHERE user_id = %s',
    }[scope]

    def settled() -> tuple[float, object]:
        rows: Final = read_rows(query, (identifier,), database_url=proxies.database)
        stored: Final = rows[0]["spend"]
        current: Final = proxies.redis.get(counter)
        assert isinstance(stored, (int, float)) and isinstance(current, bytes)
        return float(current), stored

    total: Final = spend + expected_cost
    assert eventually(settled, lambda value: value == pytest.approx((total, total)), seconds=30) == pytest.approx(
        (total, total)
    )
    return response.status_code


@pytest.mark.parametrize("scope", ["end-user", "project", "org"])
@pytest.mark.parametrize("over", [False, True], ids=["under", "over"])
def test_affiliation_lifetime_budgets(proxies: Proxies, actual_cost: float, scope: str, over: bool) -> None:
    spend: Final = 1.5 if over else 0.25
    cost: Final = 0.0 if over else actual_cost
    with _paused_between_sweeps(proxies.rust_pid, proxies.redis):
        python: Final = _affiliation_lifetime_outcome(proxies, proxies.python, scope, spend, cost)
    with _paused_between_sweeps(proxies.python_pid, proxies.redis):
        rust: Final = _affiliation_lifetime_outcome(proxies, proxies.rust, scope, spend, cost)
    assert rust == python


@pytest.mark.parametrize("allowed_model", ["metered", "budget-primary"])
def test_member_budget_allowed_models_constrain_routing(proxies: Proxies, allowed_model: str) -> None:
    identity: Final = _identity(proxies.database, Case())
    with psycopg.connect(proxies.database) as connection:
        connection.execute(
            'UPDATE "LiteLLM_BudgetTable" SET allowed_models = %s WHERE budget_id = %s',
            ([allowed_model], f"budget-{identity.user}"),
        )
    _provider_response(proxies, identity)
    python: Final = _send(proxies.python, identity)
    rust: Final = _send(proxies.rust, identity)
    if allowed_model == "metered":
        assert python.status_code == rust.status_code == 200, (python.text, rust.text)
    else:
        assert python.status_code == 403, python.text
        assert rust.status_code == 501, rust.text
        assert rust.json()["error"]["code"] == "credential"


def _model_budget_outcome(
    client: httpx.Client,
    proxies: Proxies,
    seeded: tuple[Identity, str | None, str],
    expected_spend: float,
    model: str = "metered",
) -> tuple[int, int, float, float]:
    identity, end_user, counter = seeded
    _provider_response(proxies, identity)
    first: Final = _send(client, identity, end_user, model)
    assert first.status_code == 200, first.text

    def spend() -> float:
        if proxies.uses_redis:
            value: Final = proxies.redis.get(counter)
            return float(value) if isinstance(value, bytes) else 0.0
        rows: Final = read_rows(
            """SELECT spend FROM "LiteLLM_BudgetWindowSpend"
               WHERE entity_type = 'model_max_budget' AND entity_id = %s AND window_duration = '1h'""",
            (counter,),
            database_url=proxies.database,
        )
        stored: Final = rows[0]["spend"] if rows else 0
        assert isinstance(stored, (int, float))
        return float(stored)

    settled: Final = eventually(spend, lambda value: value == pytest.approx(expected_spend))
    second: Final = _send(client, identity, end_user, model)
    assert second.status_code == 422, second.text
    assert spend() == pytest.approx(settled)
    persisted: Final = eventually(
        lambda: identity.persisted(proxies.database), lambda value: value == pytest.approx(expected_spend), seconds=30
    )
    return first.status_code, second.status_code, settled, persisted


@pytest.mark.parametrize("proxies", [True, False], ids=["redis", "postgres"], indirect=True)
def test_per_model_glob_charges_a_success_and_refuses_the_next_request(proxies: Proxies, actual_cost: float) -> None:
    python: Final = _model_budget_outcome(
        proxies.python, proxies, _model_budget_identity(proxies, actual_cost / 2, "key", "met*"), actual_cost
    )
    rust: Final = _model_budget_outcome(
        proxies.rust, proxies, _model_budget_identity(proxies, actual_cost / 2, "key", "met*"), actual_cost
    )
    assert rust == pytest.approx(python)


@pytest.mark.parametrize("scope", ["owner", "team", "member", "default", "end-user"])
def test_affiliation_per_model_caps(proxies: Proxies, actual_cost: float, scope: str) -> None:
    python_identity: Final = _model_budget_identity(proxies, actual_cost / 2, scope, "met*")
    rust_identity: Final = _model_budget_identity(proxies, actual_cost / 2, scope, "met*")
    python: Final = _model_budget_outcome(proxies.python, proxies, python_identity, actual_cost)
    rust: Final = _model_budget_outcome(proxies.rust, proxies, rust_identity, actual_cost)
    assert rust == pytest.approx(python)


@pytest.mark.parametrize(
    ("model", "entry"),
    [
        pytest.param("metered-alias", "metered", id="alias"),
        pytest.param("metered", "metered-family", id="access-group"),
    ],
)
def test_affiliation_per_model_cap_resolves_budget_name(
    proxies: Proxies, actual_cost: float, model: str, entry: str
) -> None:
    python_identity: Final = _model_budget_identity(proxies, actual_cost / 2, "owner", entry)
    rust_identity: Final = _model_budget_identity(proxies, actual_cost / 2, "owner", entry)
    python: Final = _model_budget_outcome(proxies.python, proxies, python_identity, actual_cost, model)
    rust: Final = _model_budget_outcome(proxies.rust, proxies, rust_identity, actual_cost, model)
    assert rust == pytest.approx(python)


@pytest.mark.parametrize("scope", ["key", "member"])
@pytest.mark.parametrize("allowed", [True, False], ids=["allowed", "denied"])
def test_per_model_budget_fallback_validates_target(
    proxies: Proxies, actual_cost: float, scope: str, allowed: bool
) -> None:
    def outcome(client: httpx.Client) -> tuple[int, str | None]:
        identity, _, counter = _model_budget_identity(proxies, actual_cost / 2, scope, "budget-primary")
        proxies.redis.set(counter, actual_cost)
        hashed: Final = hashlib.sha256(identity.token.encode()).hexdigest()
        with psycopg.connect(proxies.database) as connection:
            connection.execute(
                'UPDATE "LiteLLM_VerificationToken" SET budget_fallbacks = %s, models = %s WHERE token = %s',
                (
                    Jsonb({"budget-primary": ["budget-secondary-alias"]} if scope == "key" else {}),
                    ["budget-primary", "budget-secondary", "budget-secondary-alias"] if allowed else ["budget-primary"],
                    hashed,
                ),
            )
        _provider_response(proxies, identity)
        response: Final = _send(client, identity, model="budget-primary")
        if allowed:
            assert response.status_code == 200, response.text
            assert response.headers["x-litellm-model-id"] == "budget-secondary-id"
            eventually(lambda: identity.persisted(proxies.database), lambda value: value == pytest.approx(actual_cost))
        else:
            assert response.status_code == 422, response.text
            assert identity.persisted(proxies.database) == pytest.approx(0.0)
        assert float(proxies.redis.get(counter)) == pytest.approx(actual_cost)
        return response.status_code, response.headers.get("x-litellm-model-id")

    assert outcome(proxies.rust) == outcome(proxies.python)


def _outcome(client: httpx.Client, proxies: Proxies, case: Case, actual_cost: float) -> tuple[int, float, float]:
    identity: Final = _identity(proxies.database, case)
    _provider_response(proxies, identity)
    proxies.redis.set(identity.counter, case.spend)
    response: Final = _send(client, identity)
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


def test_standalone_evicts_project_and_membership_after_applying_spend(proxies: Proxies) -> None:
    identity: Final = _identity(proxies.database, Case())
    assert identity.team is not None
    project: Final = uuid.uuid4().hex
    keys: Final = (f"project_id:{project}", f"team_membership:{identity.user}:{identity.team}")
    charge: Final = 0.125
    process: Final = psutil.Process(proxies.python_pid)
    process.suspend()
    try:
        eventually(process.status, lambda status: status == psutil.STATUS_STOPPED)
        with psycopg.connect(proxies.database) as connection:
            connection.execute(
                """
                INSERT INTO "LiteLLM_ProjectTable" (project_id, created_by, updated_by)
                VALUES (%s, 'parity', 'parity')
                """,
                (project,),
            )
            assert proxies.redis.mset(dict.fromkeys(keys, b'{"spend":0}'))
            assert proxies.redis.exists(*keys) == len(keys)
            connection.execute(
                """
                INSERT INTO "LiteLLM_SpendLogDelivery" (delivery_id, request_id, is_batch, accounting_payload)
                VALUES (%s, %s, false, %s)
                """,
                (
                    project,
                    project,
                    Jsonb(
                        {
                            "spend": charge,
                            "metadata": {
                                "accounting_entities": {
                                    "project_id": project,
                                    "user": identity.user,
                                    "team_id": identity.team,
                                }
                            },
                        }
                    ),
                ),
            )

        assert eventually(lambda: proxies.redis.exists(*keys), lambda count: count == 0) == 0
        assert identity.persisted(proxies.database) == pytest.approx(charge)
        assert read_rows(
            'SELECT spend FROM "LiteLLM_ProjectTable" WHERE project_id = %s', (project,), database_url=proxies.database
        ) == [{"spend": charge}]
    finally:
        process.resume()
