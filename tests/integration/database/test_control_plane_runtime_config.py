import asyncio
import json
import os
import shutil
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Final
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from pydantic import BaseModel, ConfigDict, SecretStr
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import event, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import CreateSchema, DropSchema

from litellm.proxy.control_plane.app import create_app
from litellm.proxy.control_plane.auth.jwt import SessionTokenCodec
from litellm.proxy.control_plane.auth.rbac import (
    ADMIN_ROLE,
    VIEWER_ROLE,
    RoleDefinition,
    RolePolicy,
    Scope,
    build_policy,
    build_principal,
)
from litellm.proxy.control_plane.container import build_container
from litellm.proxy.control_plane.settings import ControlPlaneSettings
from litellm.proxy.control_plane.store.migrate import (
    ADVISORY_LOCK,
    VERSION_TABLE,
    database_url,
    engine_for,
    migrate,
    migration_config,
)
from litellm.proxy.control_plane.store.models import Change, Entry, revision_key
from litellm.proxy.control_plane.store.postgres import PostgresStore
from litellm.proxy.control_plane.store.schema import dataplanes, metadata
from litellm.proxy.control_plane.store.stream import ConfigStream, RedisStream
from litellm.types.proxy.control_plane.fleet import FleetTeamStatus, ManagedFleetTeam

MASTER_KEY: Final = "01234567890123456789012345678901"


class Licensed:
    def is_premium(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class Services:
    url: SecretStr
    redis_url: SecretStr
    stream_name: str


@pytest_asyncio.fixture
async def services() -> AsyncIterator[Services]:
    harness_managed: Final = "INTEGRATION_RUN_ID" in os.environ
    url: Final = os.environ.get("CONTROL_PLANE_TEST_DATABASE_URL") or (
        os.environ["DATABASE_URL"] if harness_managed else None
    )
    if url is None:
        pytest.skip("Set CONTROL_PLANE_TEST_DATABASE_URL to a dedicated disposable Postgres database.")
    redis_host: Final = os.environ.get("REDIS_HOST", "127.0.0.1") if harness_managed else "127.0.0.1"
    redis_port: Final = os.environ.get("REDIS_PORT", "6379") if harness_managed else "6379"
    redis_url: Final = os.environ.get("CONTROL_PLANE_TEST_REDIS_URL", f"redis://{redis_host}:{redis_port}/0")
    engine: Final = engine_for(SecretStr(url))
    redis: Final = Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
    schema: Final = "cp_test_" + uuid4().hex
    stream_name: Final = "litellm:control_plane:test:" + uuid4().hex
    try:
        async with engine.connect() as probe:
            await probe.execute(text("SELECT 1"))
        await redis.ping()
    except (SQLAlchemyError, RedisError, OSError) as error:
        await redis.aclose()
        await engine.dispose()
        if harness_managed:
            raise
        pytest.skip(f"Local control-plane services unavailable ({type(error).__name__}).")
    try:
        async with engine.begin() as connection:
            await connection.execute(CreateSchema(schema))
        isolated: Final = make_url(database_url(SecretStr(url))).update_query_dict(
            {"options": "-csearch_path=" + schema}
        )
        yield Services(SecretStr(isolated.render_as_string(hide_password=False)), SecretStr(redis_url), stream_name)
    finally:
        await redis.delete(stream_name)
        await redis.aclose()
        async with engine.begin() as cleanup:
            await cleanup.execute(DropSchema(schema, cascade=True, if_exists=True))
        await engine.dispose()


@pytest.mark.asyncio
async def test_migrations_are_idempotent_schema_version_matches_and_metadata_has_no_autogenerate_diff(
    services: Services,
) -> None:
    store: Final = PostgresStore.build(services.url)
    try:
        assert not await store.current()
        async with store.engine.connect() as empty:
            assert await empty.run_sync(lambda sync: inspect(sync).get_table_names()) == []
        await asyncio.to_thread(migrate, services.url)
        assert await store.current()
        await asyncio.to_thread(migrate, services.url)
        assert await store.current()
        async with store.engine.connect() as connection:
            differences: Final = await connection.run_sync(
                lambda sync: compare_metadata(
                    MigrationContext.configure(sync, opts={"version_table": VERSION_TABLE}), metadata
                )
            )
            for _ in range(8):
                assert await connection.scalar(text("SELECT CAST(:number AS integer)"), {"number": 1}) == 1
            assert await connection.scalar(text("SELECT COUNT(*) FROM pg_prepared_statements")) == 0
        assert differences == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_pending_revisions_run_in_order_and_each_commits_independently(
    services: Services, tmp_path: Path
) -> None:
    config: Final = migration_config()
    source: Final = config.get_main_option("script_location")
    assert source is not None
    scripts: Final = tmp_path / "migrations"
    shutil.copytree(source, scripts)
    (scripts / "versions" / "0003_probe.py").write_text(
        'from alembic import op\nrevision = "0003"\ndown_revision = "0002"\n'
        'def upgrade():\n    op.execute("CREATE TABLE ordering_probe AS SELECT COUNT(*) AS count FROM control_plane_roles")\n'
    )
    (scripts / "versions" / "0004_failure.py").write_text(
        'from alembic import op\nrevision = "0004"\ndown_revision = "0003"\n'
        'def upgrade():\n    op.execute("CREATE TABLE rolled_back (value TEXT)")\n    raise RuntimeError("planned migration failure")\n'
    )
    config.set_main_option("script_location", str(scripts))
    config.attributes["database_url"] = services.url
    with pytest.raises(RuntimeError, match="planned migration failure"):
        await asyncio.to_thread(command.upgrade, config, "head")
    store: Final = PostgresStore.build(services.url)
    try:
        async with store.engine.connect() as connection:
            assert await connection.scalar(text("SELECT count FROM ordering_probe")) == 0
            assert await connection.scalar(text("SELECT version_num FROM control_plane_alembic_version")) == "0003"
            assert not await connection.run_sync(lambda sync: inspect(sync).has_table("rolled_back"))
        assert not await store.current()
    finally:
        await store.close()


async def eventually(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(30):
        while not predicate():
            await asyncio.sleep(0)


class _RegionalBudget(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_budget: float | None = None
    budget_duration: str | None = "7d"


@dataclass
class _RegionalTeams:
    budgets: dict[str, _RegionalBudget] = field(
        default_factory=lambda: {"east.test": _RegionalBudget(), "west.test": _RegionalBudget()}
    )
    down: set[str] = field(default_factory=set)
    read_failures: dict[str, tuple[int, dict[str, object]]] = field(default_factory=dict)
    write_failures: dict[str, int | type[httpx.HTTPError]] = field(default_factory=dict)
    writes: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    team_ids: dict[str, str] = field(default_factory=dict)
    pause_after_commit: bool = False
    committed: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    slow_read: str | None = None
    slow_read_started: asyncio.Event = field(default_factory=asyncio.Event)
    release_read: asyncio.Event = field(default_factory=asyncio.Event)
    west_read_started: asyncio.Event = field(default_factory=asyncio.Event)
    slow_write: str | None = None
    slow_write_started: asyncio.Event = field(default_factory=asyncio.Event)
    release_write: asyncio.Event = field(default_factory=asyncio.Event)

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: str,
        *,
        params: Sequence[tuple[str, str]] | None = None,
        json: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float,
    ) -> AsyncIterator[httpx.Response]:
        request: Final = httpx.Request(method, url, params=params, headers=headers)
        host: Final = request.url.host
        path: Final = request.url.path
        region: Final = host.removeprefix("new-").removesuffix(".test")
        assert request.headers["Authorization"] == f"Bearer sk-{region}"
        if host in self.down:
            raise httpx.ConnectError("regional proxy stopped", request=request)
        if path == "/v2/team/list":
            assert method == "GET"
            assert request.url.params["team_alias"] == "engineering"
            assert request.url.params["sort_by"] == "team_id"
            yield httpx.Response(
                200,
                json={
                    "teams": [
                        {"team_id": f"{region}-unrelated", "team_alias": "engineering-tools"},
                        {"team_id": self.team_ids.get(host, f"{region}-team"), "team_alias": "engineering"},
                    ],
                    "total_pages": 1,
                },
                request=request,
            )
        elif path == "/team/info":
            assert method == "GET"
            assert request.url.params["team_id"] == f"{region}-team"
            if host == self.slow_read:
                self.slow_read_started.set()
                await self.release_read.wait()
            if host == "west.test":
                self.west_read_started.set()
            status, body = self.read_failures.get(host, (200, {"team_info": self.budgets[host].model_dump()}))
            yield httpx.Response(status, json=body, request=request)
        else:
            assert path == "/team/update" and method == "POST" and json is not None
            assert json["team_id"] == f"{region}-team"
            failure: Final = self.write_failures.get(host)
            if isinstance(failure, int):
                yield httpx.Response(failure, request=request)
                return
            if failure is not None:
                raise failure("regional write failed", request=request)
            if host == self.slow_write:
                self.slow_write_started.set()
                await self.release_write.wait()
            self.writes.append((host, json))
            self.budgets[host] = _RegionalBudget.model_validate({**self.budgets[host].model_dump(), **json})
            if self.pause_after_commit:
                self.committed.set()
                await self.release.wait()
            yield httpx.Response(200, json={"team_id": f"{region}-team"}, request=request)


def _fleet_settings(services: Services) -> ControlPlaneSettings:
    return ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": MASTER_KEY,
            "database_url": services.url,
            "redis_url": services.redis_url,
            "CONTROL_PLANE_SALT_KEY": "integration-encryption-key",
            "config_stream": services.stream_name,
            "audit_sinks": (),
            "dataplane_retry_attempts": 1,
            "dataplanes": {
                "east": {"url": "https://east.test", "service_token": {"token": "sk-east"}},
                "west": {"url": "https://west.test", "service_token": {"token": "sk-west"}},
            },
        }
    )


@asynccontextmanager
async def _fleet_replica(settings: ControlPlaneSettings, edge: _RegionalTeams) -> AsyncIterator[httpx.AsyncClient]:
    app: Final = create_app(build_container(settings, transport=edge), Licensed())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://control",
            headers={"Authorization": f"Bearer {MASTER_KEY}"},
        ) as client,
    ):
        yield client


_FLEET: Final = "/api/v1/fleet/teams"
_ADOPT: Final = {"team_alias": "engineering", "dataplanes": ["east", "west"]}


async def _adopt(client: httpx.AsyncClient) -> str:
    response: Final = await client.post(f"{_FLEET}/adopt", json=_ADOPT)
    assert response.status_code == 200, response.text
    return f"{_FLEET}/{ManagedFleetTeam.model_validate(response.json()).fleet_team_id}"


async def _put(client: httpx.AsyncClient, path: str, budget: float) -> ManagedFleetTeam:
    response: Final = await client.put(path, json={"max_budget": budget, "budget_duration": "7d"})
    assert response.status_code == 200, response.text
    return ManagedFleetTeam.model_validate(response.json())


async def _wait_status(client: httpx.AsyncClient, path: str, revision: str, states: tuple[str, ...]) -> FleetTeamStatus:
    async with asyncio.timeout(20):
        while True:
            response: Final = await client.get(f"{path}/status")
            assert response.status_code == 200, response.text
            status: Final = FleetTeamStatus.model_validate(response.json())
            if tuple(target.state for target in status.dataplanes) == states and all(
                target.desired_revision == revision for target in status.dataplanes
            ):
                return status
            await asyncio.sleep(0.025)


@pytest.mark.asyncio
async def test_admin_can_adopt_and_update_managed_team_only_with_access_to_all_targets(services: Services) -> None:
    await asyncio.to_thread(migrate, services.url)
    edge: Final = _RegionalTeams()
    settings: Final = _fleet_settings(services)
    policy: Final = build_policy({})
    assert isinstance(policy, RolePolicy)
    codec: Final = SessionTokenCodec(MASTER_KEY, settings.session_issuer, timedelta(minutes=5), policy)
    admin: Final = build_principal(policy, "admin", None, frozenset({ADMIN_ROLE}), None, frozenset(), "session")
    restricted_admin: Final = build_principal(
        policy, "restricted-admin", None, frozenset({ADMIN_ROLE}), None, frozenset({"east"}), "session"
    )
    headers: Final = {"Authorization": f"Bearer {codec.mint(restricted_admin)}"}
    async with _fleet_replica(settings, edge) as client:
        client.headers["Authorization"] = f"Bearer {codec.mint(admin)}"
        path: Final = await _adopt(client)
        for lookup in (path, f"{_FLEET}/missing"):
            read: Final = await client.get(f"{lookup}/status", headers=headers)
            write: Final = await client.put(lookup, json={"max_budget": 1}, headers=headers)
            assert (read.status_code, read.json()) == (write.status_code, write.json())
            assert read.status_code == 404, read.text
        assert edge.writes == []
        changed: Final = await _put(client, path, 10)
        status: Final = await _wait_status(client, path, changed.revision, ("applied", "applied"))
        assert status.desired is not None and status.desired.max_budget == 10


@pytest.mark.asyncio
async def test_fleet_team_recovers_an_offline_dataplane_and_reports_local_drift(services: Services) -> None:
    await asyncio.to_thread(migrate, services.url)
    edge: Final = _RegionalTeams()
    async with _fleet_replica(_fleet_settings(services), edge) as client:
        edge.down.add("west.test")
        refused: Final = await client.post(f"{_FLEET}/adopt", json=_ADOPT)
        assert refused.status_code == 409, refused.text
        assert refused.json()["conflicts"] == [
            {"dataplane_id": "west", "reason": "unreachable after 1 attempt(s): ConnectError"}
        ]
        edge.down.clear()
        path: Final = await _adopt(client)
        duplicate: Final = await client.post(f"{_FLEET}/adopt", json=_ADOPT)
        assert duplicate.status_code == 409, duplicate.text
        assert sorted(conflict["dataplane_id"] for conflict in duplicate.json()["conflicts"]) == ["east", "west"]
        edge.down.add("west.test")
        changed: Final = await _put(client, path, 0)
        partial: Final = await _wait_status(client, path, changed.revision, ("applied", "unreachable"))
        assert partial.dataplanes[0].applied_revision == changed.revision
        assert partial.dataplanes[1].applied_revision is None
        assert partial.dataplanes[1].last_error == "unreachable after 1 attempt(s): ConnectError"
        edge.down.clear()
        recovered: Final = await _wait_status(client, path, changed.revision, ("applied", "applied"))
        assert recovered.dataplanes[1].applied_revision == changed.revision
        assert recovered.dataplanes[1].last_error is None
        edge.budgets["east.test"] = _RegionalBudget(max_budget=90)
        drifted: Final = await _wait_status(client, path, changed.revision, ("drifted", "applied"))
        assert drifted.dataplanes[0].applied_revision == changed.revision
        assert edge.budgets["east.test"].max_budget == 90
        assert edge.writes == [
            ("east.test", {"team_id": "east-team", "max_budget": 0}),
            ("west.test", {"team_id": "west-team", "max_budget": 0}),
        ]


@pytest.mark.asyncio
async def test_fleet_budget_quarantines_bad_committed_rows(services: Services) -> None:
    await asyncio.to_thread(migrate, services.url)
    edge: Final = _RegionalTeams()
    settings: Final = _fleet_settings(services).model_copy(
        update={"dataplanes": {"east": _fleet_settings(services).dataplanes["east"]}}
    )
    policy: Final = build_policy({"budget-editor": RoleDefinition(frozenset({Scope.FLEET_WRITE}), frozenset())})
    assert isinstance(policy, RolePolicy)
    codec: Final = SessionTokenCodec(MASTER_KEY, settings.session_issuer, timedelta(minutes=5), policy)
    editor: Final = build_principal(
        policy, "editor", None, frozenset({"budget-editor", VIEWER_ROLE}), None, frozenset(), "session"
    )
    headers: Final = {"Authorization": f"Bearer {codec.mint(editor)}"}
    store: Final = PostgresStore.build(services.url)
    try:
        async with _fleet_replica(settings, edge) as client:
            registered: Final = await client.put(
                "/api/v1/dataplanes/west",
                json={"url": "https://west.test", "service_token": {"token": "sk-west"}},
            )
            assert registered.status_code == 200, registered.text
            role: Final = await client.put("/api/v1/roles/budget-editor", json={"scopes": ["fleet:write"]})
            assert role.status_code == 200, role.text
            path: Final = await _adopt(client)
            async with store.transaction() as transaction:
                await transaction.write(
                    Entry(
                        revision="1-0",
                        change=Change(
                            op="upsert",
                            kind="role",
                            name="budget-editor",
                            actor="test",
                            entity={"scopes": ["fleet:write", "invalid:scope"]},
                        ),
                    )
                )
                await transaction.write(
                    Entry(
                        revision="2-0",
                        change=Change(
                            op="upsert",
                            kind="dataplane",
                            name="west",
                            actor="test",
                            entity={"url": "https://west.test", "service_token": {}},
                        ),
                    )
                )
            changed: Final = await _put(client, path, 25)
            denied: Final = await client.put(path, json={"max_budget": 90}, headers=headers)
            assert denied.status_code == 403, denied.text
            status: Final = await _wait_status(client, path, changed.revision, ("applied", "unreachable"))
            assert status.dataplanes[1].applied_revision is None
            assert status.dataplanes[1].last_error == "Dataplane is missing or its registered URL changed."
            assert edge.writes == [("east.test", {"team_id": "east-team", "max_budget": 25})]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_fleet_team_readoption_repairs_only_the_same_regional_team(services: Services) -> None:
    await asyncio.to_thread(migrate, services.url)
    edge: Final = _RegionalTeams(team_ids={"new-west.test": "different-team"})
    edge.budgets["new-west.test"] = _RegionalBudget()
    settings: Final = _fleet_settings(services).model_copy(update={"dataplanes": {}})
    async with _fleet_replica(settings, edge) as client:
        for region in ("east", "west"):
            registered: Final = await client.put(
                f"/api/v1/dataplanes/{region}",
                json={"url": f"https://{region}.test", "service_token": {"token": f"sk-{region}"}},
            )
            assert registered.status_code == 200, registered.text
        path: Final = await _adopt(client)
        changed: Final = await _put(client, path, 25)
        await _wait_status(client, path, changed.revision, ("applied", "applied"))
        moved: Final = await client.put(
            "/api/v1/dataplanes/west",
            json={"url": "https://new-west.test", "service_token": {"token": "sk-west"}},
        )
        assert moved.status_code == 200, moved.text
        blocked: Final = await _wait_status(client, path, changed.revision, ("applied", "unreachable"))
        assert blocked.dataplanes[1].last_error == "Dataplane is missing or its registered URL changed."
        payload: Final = {"team_alias": "engineering", "dataplanes": ["west"], "fleet_team_id": changed.fleet_team_id}
        refused: Final = await client.post(f"{_FLEET}/adopt", json=payload)
        assert refused.status_code == 409, refused.text
        assert refused.json()["conflicts"] == [{"dataplane_id": "west", "reason": "Existing team link does not match."}]
        assert sorted(host for host, _ in edge.writes) == ["east.test", "west.test"]
        edge.team_ids.clear()
        repaired: Final = await client.post(f"{_FLEET}/adopt", json=payload)
        assert repaired.status_code == 200, repaired.text
        assert ManagedFleetTeam.model_validate(repaired.json()) == changed
        await _wait_status(client, path, changed.revision, ("applied", "applied"))
        repeated: Final = await client.post(f"{_FLEET}/adopt", json=payload)
        assert repeated.status_code == 200, repeated.text
        assert repeated.json() == repaired.json()
        assert sorted(host for host, _ in edge.writes) == ["east.test", "new-west.test", "west.test"]
        assert edge.budgets["new-west.test"].max_budget == 25


@pytest.mark.asyncio
@pytest.mark.parametrize("already_applied", [False, True])
@pytest.mark.parametrize(
    ("status_code", "body", "reason"),
    [
        (403, {}, "HTTP 403 with the service_token credential"),
        (404, {}, "HTTP 404 with the service_token credential"),
        (302, {"team_info": {"max_budget": 25, "budget_duration": "7d"}}, "HTTP 302 with the service_token credential"),
        (200, {}, "unexpected response shape: 1 validation error(s)"),
    ],
)
async def test_fleet_status_reports_regional_read_failures_as_pending(
    services: Services, status_code: int, body: dict[str, object], reason: str, already_applied: bool
) -> None:
    await asyncio.to_thread(migrate, services.url)
    edge: Final = _RegionalTeams(read_failures={} if already_applied else {"west.test": (status_code, body)})
    async with _fleet_replica(_fleet_settings(services), edge) as client:
        path: Final = await _adopt(client)
        changed: Final = await _put(client, path, 25)
        if already_applied:
            await _wait_status(client, path, changed.revision, ("applied", "applied"))
            edge.read_failures["west.test"] = (status_code, body)
        async with asyncio.timeout(10):
            while True:
                reply = await client.get(f"{path}/status")  # rebind-ok: poll for the regional read failure
                assert reply.status_code == 200, reply.text
                west = FleetTeamStatus.model_validate(reply.json()).dataplanes[1]  # rebind-ok: refresh west status
                if west.last_error is not None:
                    break
                await asyncio.sleep(0.025)
        assert west.dataplane_id == "west"
        assert west.desired_revision == changed.revision
        assert west.applied_revision == (changed.revision if already_applied else None)
        assert (west.state, west.last_error) == ("pending", reason)
        assert sum(host == "west.test" for host, _ in edge.writes) == int(already_applied)
        if not already_applied:
            audit: Final = await client.get("/api/v1/audit", params={"dataplane": "west"})
            assert audit.status_code == 200, audit.text
            assert {item["decision"] for item in audit.json()["items"] if (item["detail"] or "").endswith(reason)} == {
                "denied"
            }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "state", "reason"),
    [
        (httpx.ConnectError, "unreachable", "unreachable after 1 attempt(s): ConnectError"),
        (httpx.ReadTimeout, "unreachable", "unreachable after 1 attempt(s): ReadTimeout"),
        (403, "pending", "Dataplane budget update failed (HTTP 403)."),
    ],
)
async def test_fleet_status_distinguishes_regional_write_transport_and_http_failures(
    services: Services, failure: int | type[httpx.HTTPError], state: str, reason: str
) -> None:
    await asyncio.to_thread(migrate, services.url)
    edge: Final = _RegionalTeams(write_failures={"west.test": failure})
    async with _fleet_replica(_fleet_settings(services), edge) as client:
        path: Final = await _adopt(client)
        changed: Final = await _put(client, path, 25)
        async with asyncio.timeout(10):
            while True:
                reply = await client.get(f"{path}/status")  # rebind-ok: poll for the regional write failure
                assert reply.status_code == 200, reply.text
                west = FleetTeamStatus.model_validate(reply.json()).dataplanes[1]  # rebind-ok: refresh west status
                if west.last_error is not None:
                    break
                await asyncio.sleep(0.025)
        assert west.dataplane_id == "west"
        assert west.desired_revision == changed.revision
        assert west.applied_revision is None
        assert (west.state, west.last_error) == (state, reason)


@pytest.mark.asyncio
async def test_concurrent_fleet_team_puts_converge_to_the_last_committed_stream_revision(services: Services) -> None:
    await asyncio.to_thread(migrate, services.url)
    edge: Final = _RegionalTeams()
    settings: Final = _fleet_settings(services)
    async with _fleet_replica(settings, edge) as first, _fleet_replica(settings, edge) as second:
        path: Final = await _adopt(first)
        earlier, later = sorted(
            await asyncio.gather(_put(first, path, 25), _put(second, path, 50)),
            key=lambda response: revision_key(response.revision),
        )
        status: Final = await _wait_status(first, path, later.revision, ("applied", "applied"))
        assert earlier.revision != later.revision
        assert status.desired == later.desired
        assert tuple(target.applied_revision for target in status.dataplanes) == (later.revision, later.revision)
        assert later.desired is not None
        assert edge.budgets == {
            "east.test": _RegionalBudget(max_budget=later.desired.max_budget),
            "west.test": _RegionalBudget(max_budget=later.desired.max_budget),
        }


def _tracked_fleet_settings(services: Services) -> tuple[ControlPlaneSettings, str]:
    app_name: Final = f"fleet_put_{uuid4().hex}"
    app_url: Final = make_url(database_url(services.url)).update_query_dict({"application_name": app_name})
    settings: Final = _fleet_settings(services).model_copy(
        update={"database_url": SecretStr(app_url.render_as_string(hide_password=False))}
    )
    return settings, app_name


async def _wait_for_put_row_lock(services: Services, app_name: str, put: asyncio.Task[ManagedFleetTeam]) -> None:
    engine: Final = engine_for(services.url)
    try:
        async with engine.connect() as probe, asyncio.timeout(10):
            while not put.done():
                waiting = await probe.scalar(  # rebind-ok: poll for the PUT row lock
                    text(
                        "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE application_name = :app "
                        "AND wait_event_type = 'Lock' AND (query LIKE 'UPDATE control_plane_fleet_teams SET%' "
                        "OR query LIKE '%FROM control_plane_fleet_teams%FOR UPDATE'))"
                    ),
                    {"app": app_name},
                )
                await probe.commit()
                if waiting:
                    return
                await asyncio.sleep(0.025)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_fleet_team_put_waits_for_inflight_regional_write(services: Services) -> None:
    await asyncio.to_thread(migrate, services.url)
    settings, app_name = _tracked_fleet_settings(services)
    edge: Final = _RegionalTeams(slow_write="east.test")
    async with _fleet_replica(settings, edge) as client:
        path: Final = await _adopt(client)
        await _put(client, path, 25)
        await asyncio.wait_for(edge.slow_write_started.wait(), timeout=10)
        later: Final = asyncio.create_task(_put(client, path, 50))
        try:
            await _wait_for_put_row_lock(services, app_name, later)
            put_committed_before_release: Final = later.done()
            assert edge.budgets["east.test"].max_budget is None
        finally:
            edge.release_write.set()
            latest: Final = await asyncio.wait_for(later, timeout=10)

        await _wait_status(client, path, latest.revision, ("applied", "applied"))
        east_budgets: Final = [body["max_budget"] for host, body in edge.writes if host == "east.test"]
        assert east_budgets == [25, 50]
        assert not put_committed_before_release, f"PUT committed before east received stale budget {east_budgets[0]}"


@pytest.mark.asyncio
async def test_waiting_fleet_team_put_does_not_block_status_or_config_writes(services: Services) -> None:
    await asyncio.to_thread(migrate, services.url)
    settings, app_name = _tracked_fleet_settings(services)
    edge: Final = _RegionalTeams(slow_write="east.test")
    async with _fleet_replica(settings, edge) as client:
        path: Final = await _adopt(client)
        initial: Final = await _put(client, path, 25)
        await asyncio.wait_for(edge.slow_write_started.wait(), timeout=10)
        later: Final = asyncio.create_task(_put(client, path, 50))
        try:
            await _wait_for_put_row_lock(services, app_name, later)
            assert not later.done()
            status, role = await asyncio.wait_for(
                asyncio.gather(
                    client.get(f"{path}/status"),
                    client.put("/api/v1/roles/unrelated-reader", json={"scopes": ["usage:read"]}),
                ),
                timeout=5,
            )
            assert status.status_code == 200, status.text
            assert FleetTeamStatus.model_validate(status.json()).revision == initial.revision
            assert role.status_code == 200, role.text
            assert not later.done()
        finally:
            edge.release_write.set()
            latest: Final = await asyncio.wait_for(later, timeout=10)
        await _wait_status(client, path, latest.revision, ("applied", "applied"))


@pytest.mark.asyncio
async def test_fleet_worker_retries_and_logs_unexpected_store_errors(
    services: Services, caplog: pytest.LogCaptureFixture
) -> None:
    await asyncio.to_thread(migrate, services.url)
    edge: Final = _RegionalTeams()
    container: Final = build_container(_fleet_settings(services), transport=edge)
    store: Final = container.runtime_config.store
    assert isinstance(store, PostgresStore)
    faults: Final = {
        "SELECT control_plane_fleet_team_dataplanes.fleet_team_id,": "pending query failed once",
        "FOR UPDATE OF control_plane_fleet_team_dataplanes SKIP LOCKED": "mapping claim failed once",
    }

    @event.listens_for(store.engine.sync_engine, "before_cursor_execute")
    def fail_once(_connection: object, _cursor: object, statement: str, *_args: object) -> None:
        match: Final = next((fragment for fragment in faults if fragment in statement), None)
        if match is not None:
            raise RuntimeError(faults.pop(match))

    app: Final = create_app(container, Licensed())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://control",
            headers={"Authorization": f"Bearer {MASTER_KEY}"},
        ) as client,
    ):
        path: Final = await _adopt(client)
        changed: Final = await _put(client, path, 45)
        await _wait_status(client, path, changed.revision, ("applied", "applied"))
        assert sorted(edge.writes, key=lambda write: write[0]) == [
            ("east.test", {"team_id": "east-team", "max_budget": 45}),
            ("west.test", {"team_id": "west-team", "max_budget": 45}),
        ]
        assert [
            str(record.exc_info[1])
            for record in caplog.records
            if record.name == "litellm.control_plane.fleet" and record.exc_info
        ] == ["pending query failed once", "mapping claim failed once"]


@pytest.mark.asyncio
async def test_fleet_worker_applies_healthy_mapping_while_another_read_is_blocked(services: Services) -> None:
    await asyncio.to_thread(migrate, services.url)
    edge: Final = _RegionalTeams(slow_read="east.test")
    async with _fleet_replica(_fleet_settings(services), edge) as client:
        path: Final = await _adopt(client)
        changed: Final = await _put(client, path, 10)
        async with asyncio.timeout(10):
            await edge.slow_read_started.wait()
            partial: Final = await _wait_status(client, path, changed.revision, ("pending", "applied"))
        assert tuple(target.applied_revision for target in partial.dataplanes) == (None, changed.revision)
        assert edge.writes == [("west.test", {"team_id": "west-team", "max_budget": 10})]
        edge.release_read.set()
        await _wait_status(client, path, changed.revision, ("applied", "applied"))


@pytest.mark.asyncio
async def test_fleet_status_reads_healthy_mapping_while_another_read_is_blocked(services: Services) -> None:
    await asyncio.to_thread(migrate, services.url)
    edge: Final = _RegionalTeams()
    async with _fleet_replica(_fleet_settings(services), edge) as client:
        path: Final = await _adopt(client)
        changed: Final = await _put(client, path, 10)
        await _wait_status(client, path, changed.revision, ("applied", "applied"))

        edge.slow_read = "east.test"
        edge.west_read_started.clear()
        pending: Final = asyncio.create_task(client.get(f"{path}/status"))
        try:
            async with asyncio.timeout(10):
                await edge.slow_read_started.wait()
                await edge.west_read_started.wait()
            assert not pending.done()
        finally:
            edge.release_read.set()
            reply: Final = await pending
        assert reply.status_code == 200, reply.text
        status: Final = FleetTeamStatus.model_validate(reply.json())
        assert [(target.dataplane_id, target.state) for target in status.dataplanes] == [
            ("east", "applied"),
            ("west", "applied"),
        ]


@pytest.mark.asyncio
async def test_fleet_team_restart_after_remote_commit_does_not_write_twice(services: Services) -> None:
    await asyncio.to_thread(migrate, services.url)
    edge: Final = _RegionalTeams(pause_after_commit=True, slow_read="west.test")
    settings: Final = _fleet_settings(services)
    async with _fleet_replica(settings, edge) as before:
        path: Final = await _adopt(before)
        changed: Final = await _put(before, path, 30)
        async with asyncio.timeout(20):
            await edge.committed.wait()
        pending_reply: Final = await before.get(f"{path}/status")
        assert pending_reply.status_code == 200, pending_reply.text
        pending: Final = FleetTeamStatus.model_validate(pending_reply.json())
        assert tuple(target.applied_revision for target in pending.dataplanes) == (None, None)
        assert edge.writes == [("east.test", {"team_id": "east-team", "max_budget": 30})]
    edge.pause_after_commit = False
    edge.release_read.set()
    async with _fleet_replica(settings, edge) as after:
        recovered: Final = await _wait_status(after, path, changed.revision, ("applied", "applied"))
        assert tuple(target.applied_revision for target in recovered.dataplanes) == (changed.revision, changed.revision)
        assert edge.writes == [
            ("east.test", {"team_id": "east-team", "max_budget": 30}),
            ("west.test", {"team_id": "west-team", "max_budget": 30}),
        ]


@pytest.mark.asyncio
async def test_migration_job_waits_for_the_postgres_advisory_lock(services: Services) -> None:
    engine: Final = engine_for(services.url)
    try:
        async with engine.begin() as holder:
            await holder.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": ADVISORY_LOCK})
            migrating: Final = asyncio.create_task(asyncio.to_thread(migrate, services.url))
            async with asyncio.timeout(30):
                async with engine.connect() as observer:
                    while True:
                        waiting: Final = await observer.scalar(
                            text(
                                "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' "
                                "AND classid = :high AND objid = :low AND NOT granted)"
                            ),
                            {"high": ADVISORY_LOCK >> 32, "low": ADVISORY_LOCK & 0xFFFFFFFF},
                        )
                        if waiting:
                            break
            assert not migrating.done()
        await migrating
        assert await PostgresStore(engine).current()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_real_postgres_redis_workers_propagate_encrypted_dataplanes_roles_and_deletes(services: Services) -> None:
    await asyncio.to_thread(migrate, services.url)
    settings: Final = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": MASTER_KEY,
            "database_url": services.url,
            "redis_url": services.redis_url,
            "CONTROL_PLANE_SALT_KEY": "integration-encryption-key",
            "config_stream": services.stream_name,
            "audit_sinks": (),
        }
    )
    first: Final = build_container(settings)
    second: Final = build_container(settings)
    redis: Final = Redis.from_url(services.redis_url.get_secret_value(), decode_responses=True)
    store: Final = PostgresStore.build(services.url)
    app_first: Final = create_app(first, Licensed())
    app_second: Final = create_app(second, Licensed())
    try:
        async with app_first.router.lifespan_context(app_first), app_second.router.lifespan_context(app_second):
            async with (
                httpx.AsyncClient(transport=httpx.ASGITransport(app=app_first), base_url="http://writer") as writer,
                httpx.AsyncClient(transport=httpx.ASGITransport(app=app_second), base_url="http://reader") as reader,
            ):
                headers: Final = {"Authorization": f"Bearer {MASTER_KEY}"}
                changed: Final = await writer.put(
                    "/api/v1/dataplanes/live",
                    headers=headers,
                    json={
                        "url": "https://runtime.test",
                        "service_token": {"token": "must-not-persist-in-plaintext"},
                    },
                )
                assert changed.status_code == 200, changed.text
                await eventually(
                    lambda: first.registry.get("live") is not None and second.registry.get("live") is not None
                )
                seen: Final = await reader.get("/api/v1/dataplanes/live", headers=headers)
                assert seen.json() == changed.json()
                async with store.engine.connect() as connection:
                    persisted: Final = await connection.scalar(
                        select(dataplanes.c.spec).where(dataplanes.c.id == "live")
                    )
                published: Final = await redis.xrange(services.stream_name)
                assert "must-not-persist-in-plaintext" not in json.dumps(persisted) + str(published)
                role: Final = await writer.put(
                    "/api/v1/roles/live-reader", headers=headers, json={"scopes": ["usage:read"]}
                )
                assert role.status_code == 200, role.text
                await eventually(lambda: "live-reader" in second.policy.names)
                assert tuple(scope.value for scope in second.policy.scopes_for("live-reader")) == ("usage:read",)
                deleted: Final = await writer.delete("/api/v1/dataplanes/live", headers=headers)
                assert deleted.status_code == 204, deleted.text
                await eventually(lambda: first.registry.get("live") is None and second.registry.get("live") is None)
                assert (await reader.get("/api/v1/dataplanes/live", headers=headers)).status_code == 404
                async with store.transaction() as transaction:
                    rows: Final = await transaction.load()
                assert tuple(row.change.name for row in rows) == ("live-reader",)
    finally:
        await redis.aclose()
        await store.close()


@dataclass(frozen=True, slots=True)
class PausedDeleteStream:
    delegate: ConfigStream
    deleting: asyncio.Event
    release: asyncio.Event

    async def append(self, change: Change) -> str:
        if change.op == "delete":
            self.deleting.set()
            await self.release.wait()
        return await self.delegate.append(change)

    async def read(self, after: str) -> tuple[Entry, ...]:
        return await self.delegate.read(after)

    async def close(self) -> None:
        await self.delegate.close()


@pytest.mark.asyncio
async def test_concurrent_role_delete_and_inherit_cannot_commit_an_orphan(services: Services) -> None:
    await asyncio.to_thread(migrate, services.url)
    settings: Final = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": MASTER_KEY,
            "database_url": services.url,
            "redis_url": services.redis_url,
            "CONTROL_PLANE_SALT_KEY": "integration-key",
            "audit_sinks": (),
        }
    )
    stream: Final = PausedDeleteStream(
        RedisStream.build(services.redis_url, services.stream_name, 100),
        asyncio.Event(),
        asyncio.Event(),
    )
    first: Final = build_container(settings, config_stream=stream)
    second: Final = build_container(settings, config_stream=stream)
    first.runtime_config.ready = second.runtime_config.ready = True
    store: Final = PostgresStore.build(services.url)
    headers: Final = {"Authorization": "Bearer " + MASTER_KEY}
    try:
        async with (
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app(first, Licensed())), base_url="http://delete"
            ) as deleting,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app(second, Licensed())), base_url="http://inherit"
            ) as inheriting,
        ):
            seeded: Final = await deleting.put("/api/v1/roles/parent", headers=headers, json={"scopes": []})
            assert seeded.status_code == 200, seeded.text
            delete: Final = asyncio.create_task(deleting.delete("/api/v1/roles/parent", headers=headers))
            async with asyncio.timeout(30):
                await stream.deleting.wait()
            child: Final = asyncio.create_task(
                inheriting.put(
                    "/api/v1/roles/child",
                    headers=headers,
                    json={"inherits": ["parent"]},
                )
            )
            try:
                async with asyncio.timeout(30):
                    async with store.engine.connect() as observer:
                        while not child.done():
                            waiting: Final = await observer.scalar(
                                text(
                                    "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' "
                                    "AND classid = :high AND objid = :low AND NOT granted)"
                                ),
                                {"high": ADVISORY_LOCK >> 32, "low": ADVISORY_LOCK & 0xFFFFFFFF},
                            )
                            if waiting:
                                break
            finally:
                stream.release.set()
            removed, refused = await asyncio.gather(delete, child)
            assert removed.status_code == 204, removed.text
            assert refused.status_code == 400, refused.text
            async with store.transaction() as transaction:
                assert await transaction.load() == ()
    finally:
        stream.release.set()
        await first.runtime_config.close()
        await second.runtime_config.close()
        await store.close()
