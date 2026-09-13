import subprocess
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from .checks import COMPLETE, assert_completed
from .containers import Containers, docker, ready, until
from .database import Database, Databases, prisma_url, restricted_user

POOL_IMAGE: Final = (
    "ghcr.io/cloudnative-pg/pgbouncer@sha256:e6ddfe22d845e603825e235dd8334b21ecd125abea2a2172478f556b8dee2bb8"
)
pytestmark: Final = [pytest.mark.e2e, pytest.mark.migration_startup]


@contextmanager
def application_user(database: Database) -> Generator[Database]:
    with restricted_user(database) as application:
        role: Final = sql.Identifier(str(urlsplit(application.url).username))
        schema: Final = sql.Identifier(database.schema)
        database.execute(sql.SQL("REVOKE CREATE ON SCHEMA {} FROM PUBLIC").format(schema))
        for statement in (
            "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {} TO {}",
            "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA {} TO {}",
            "ALTER DEFAULT PRIVILEGES IN SCHEMA {} GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {}",
            "ALTER DEFAULT PRIVILEGES IN SCHEMA {} GRANT USAGE, SELECT ON SEQUENCES TO {}",
        ):
            database.execute(sql.SQL(statement).format(schema, role))
        assert application.query("SELECT has_schema_privilege(current_user, %s, 'CREATE')", (database.schema,)) == (
            (False,),
        )
        yield application


@contextmanager
def pool(database: Database, output: Path) -> Generator[str]:
    name: Final = f"litellm-migration-pool-{uuid4().hex[:12]}"
    url: Final = urlsplit(database.container_url)
    directory: Final = output / name
    directory.mkdir(parents=True)
    (directory / "users.txt").write_text(f'"{url.username}" "{url.password}"\n')
    (directory / "pgbouncer.ini").write_text(
        f"[databases]\n* = host={url.hostname} port={url.port} user={url.username} password={url.password}\n"
        "[pgbouncer]\nlisten_addr = 0.0.0.0\nlisten_port = 6432\nauth_type = trust\nauth_file = /pool/users.txt\n"
        "pool_mode = transaction\ndefault_pool_size = 1\nreserve_pool_size = 0\nmax_client_conn = 100\n"
        "max_prepared_statements = 100\nquery_wait_timeout = 8\nignore_startup_parameters = extra_float_digits,options\n"
    )
    try:
        docker(
            "run",
            "-d",
            "--name",
            name,
            "--label",
            "litellm-migration-test=true",
            "--add-host",
            "host.docker.internal:host-gateway",
            "-p",
            "0.0.0.0::6432",
            "-v",
            f"{directory}:/pool:ro",
            "--entrypoint",
            "/usr/bin/pgbouncer",
            POOL_IMAGE,
            "/pool/pgbouncer.ini",
        )
        port: Final = int(docker("port", name, "6432/tcp").splitlines()[0].rsplit(":", 1)[1])
        local_url: Final = urlunsplit(url._replace(netloc=f"{url.username}:{url.password}@127.0.0.1:{port}"))

        def connected() -> bool:
            try:
                with psycopg.connect(local_url, autocommit=True, connect_timeout=2) as connection:
                    return connection.execute("SELECT 1").fetchone() == (1,)
            except psycopg.Error:
                return False

        until("PgBouncer ready", connected, 30)
        yield local_url.replace("127.0.0.1", "host.docker.internal") + "?pgbouncer=true"
    finally:
        try:
            logs: Final = subprocess.run(("docker", "logs", name), text=True, capture_output=True, timeout=30)
            (directory / "pool.log").write_text(logs.stdout + logs.stderr)
        finally:
            subprocess.run(("docker", "rm", "-f", name), capture_output=True, text=True, timeout=30, check=True)


class TestMigrationPooling:
    @pytest.mark.parametrize("scenario,replica_count", (("fresh", 3), ("upgrade", 3), ("legacy", 3), ("upgrade", 6)))
    def test_direct_migrations_with_one_application_backend(
        self,
        containers: Containers,
        databases: Databases,
        migrated_template: Database,
        scenario: str,
        replica_count: int,
    ) -> None:
        with databases.create(None if scenario == "fresh" else migrated_template) as database:
            if scenario == "legacy":
                database.execute("DROP TABLE _prisma_migrations")
            with (
                application_user(database) as application,
                pool(application, containers.output) as pooled_url,
                ExitStack() as stack,
            ):
                replicas: Final = tuple(
                    stack.enter_context(
                        containers.start(
                            database,
                            (COMPLETE,) if scenario == "upgrade" else (),
                            environment={
                                "DATABASE_URL": prisma_url(pooled_url, database.schema),
                                "DIRECT_URL": database.container_url,
                            },
                        )
                    )
                    for _ in range(replica_count)
                )
                ready(replicas, database)
                if scenario == "upgrade":
                    assert_completed(database)
                if scenario == "legacy":
                    assert any(
                        "historical data backfills were not replayed or verified" in replica.logs()
                        for replica in replicas
                    )
                    assert database.query("SELECT count(*) FROM _prisma_migrations WHERE applied_steps_count <> 0") == (
                        (0,),
                    )
