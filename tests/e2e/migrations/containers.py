from __future__ import annotations

import hashlib
import subprocess
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from uuid import uuid4

from e2e_http import NoBody, Success, unwrap
from models import KeyGenerateBody, KeyGenerateResponse, KeyInfoParams, KeyInfoResponse
from transport import HttpTransport

from .database import Database, prisma_url
from .startup_models import ContainerState, Migration, Observation, Readiness

MASTER_KEY: Final = "sk-migration-ci-fixture"


def docker(*args: str) -> str:
    result: Final = subprocess.run(("docker", *args), capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, f"Docker operation failed: {result.stderr}"
    return result.stdout.strip()


def until(description: str, condition: Callable[[], bool], seconds: float = 150) -> None:
    deadline: Final = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.25)
    raise AssertionError(f"Timed out waiting for {description}")


@dataclass(frozen=True, slots=True)
class Replica:
    name: str
    transport: HttpTransport
    output: Path

    def state(self) -> ContainerState:
        return ContainerState.model_validate_json(docker("inspect", "--format", "{{json .State}}", self.name))

    def observe(self) -> Observation:
        state: Final = self.state()
        result: Final = self.transport.get(
            "/health/readiness", headers=self.transport.master, params=NoBody(), response_type=Readiness, timeout=1
        )
        ready: Final = isinstance(result, Success) and result.data.status == "healthy" and result.data.db == "connected"
        return Observation(None if state.Running else state.ExitCode, ready)

    def logs(self) -> str:
        result: Final = subprocess.run(("docker", "logs", self.name), capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        return result.stdout + result.stderr

    def kill(self) -> None:
        if self.state().Running:
            docker("kill", self.name)

    def usable(self, database: Database) -> None:
        alias: Final = f"migration-{uuid4().hex}"
        key: Final = unwrap(
            self.transport.post(
                "/key/generate",
                headers=self.transport.master,
                json=KeyGenerateBody(key_alias=alias),
                response_type=KeyGenerateResponse,
            )
        ).key
        info: Final = unwrap(
            self.transport.get(
                "/key/info",
                headers=self.transport.master,
                params=KeyInfoParams(key=key),
                response_type=KeyInfoResponse,
            )
        )
        assert info.info.key_alias == alias
        assert database.query(
            'SELECT key_alias FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (hashlib.sha256(key.encode()).hexdigest(),),
        ) == ((alias,),)


def ready(replicas: tuple[Replica, ...], database: Database) -> None:
    def all_ready() -> bool:
        observations: Final = tuple(replica.observe() for replica in replicas)
        assert all(item.exit_code is None for item in observations), "Replica exited before readiness"
        return all(item.ready for item in observations)

    until("every replica ready", all_ready)
    for replica in replicas:
        replica.usable(database)


def failed(replicas: tuple[Replica, ...], marker: str) -> None:
    def all_stopped() -> bool:
        observations: Final = tuple(replica.observe() for replica in replicas)
        assert not any(item.ready for item in observations), "Failed migration exposed a ready proxy"
        return all(item.exit_code is not None for item in observations)

    until("every replica to reject startup", all_stopped)
    for replica in replicas:
        assert replica.state().ExitCode != 0, "Failed startup returned success"
        assert marker in replica.logs(), f"Startup failed outside the expected migration: {marker}"


def waiting(replicas: tuple[Replica, ...], seconds: float) -> None:
    deadline: Final = time.monotonic() + seconds
    while time.monotonic() < deadline:
        assert all(item.exit_code is None and not item.ready for item in (replica.observe() for replica in replicas)), (
            "Contending replica exited or served early"
        )
        time.sleep(0.25)


@dataclass(frozen=True, slots=True)
class Containers:
    image: str
    output: Path

    @contextmanager
    def start(
        self,
        database: Database,
        migrations: tuple[Migration, ...] = (),
        *,
        v2: bool = True,
        disabled: bool = False,
        environment: Mapping[str, str] | None = None,
    ) -> Generator[Replica]:
        name: Final = f"litellm-migration-{uuid4().hex[:16]}"
        directory: Final = self.output / name
        directory.mkdir(parents=True)
        for migration in migrations:
            write_migration(directory, migration)
        (directory / "config.yaml").write_text(
            "model_list: []\ngeneral_settings:\n  master_key: os.environ/LITELLM_MASTER_KEY\n"
        )
        env: Final = {
            "DATABASE_URL": prisma_url(database.container_url, database.schema),
            "LITELLM_MASTER_KEY": MASTER_KEY,
            "LITELLM_SALT_KEY": MASTER_KEY,
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "LITELLM_TELEMETRY": "False",
            "LITELLM_LOG": "INFO",
            "DATABASE_CONNECTION_POOL_LIMIT": "2",
            "DEFAULT_NUM_WORKERS_LITELLM_PROXY": "1",
            "USE_V2_MIGRATION_RESOLVER": str(v2).lower(),
            "DISABLE_SCHEMA_UPDATE": str(disabled).lower(),
            "LITELLM_MIGRATION_DIR": "/migration-test/prisma",
            "LITELLM_PRISMA_MIGRATE_DEPLOY_TIMEOUT": "180",
            **(environment or {}),
        }
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
                "127.0.0.1::4000",
                "-v",
                f"{directory}:/migration-test",
                *(arg for key, value in env.items() for arg in ("-e", f"{key}={value}")),
                self.image,
                "--config",
                "/migration-test/config.yaml",
                "--host",
                "0.0.0.0",
                "--port",
                "4000",
            )
            port: Final = int(docker("port", name, "4000/tcp").rsplit(":", 1)[1])
            replica: Final = Replica(name, HttpTransport(f"http://127.0.0.1:{port}", MASTER_KEY, 15), directory)
            yield replica
        finally:
            try:
                state: Final = subprocess.run(
                    ("docker", "inspect", "--format", "{{json .State}}", name),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                (directory / "state.json").write_text(state.stdout or state.stderr)
                logs: Final = subprocess.run(("docker", "logs", name), capture_output=True, text=True, timeout=30)
                (directory / "proxy.log").write_text(logs.stdout + logs.stderr)
            finally:
                subprocess.run(("docker", "rm", "-f", name), capture_output=True, text=True, timeout=30, check=True)


def write_migration(directory: Path, migration: Migration) -> None:
    path: Final = directory / "prisma" / "migrations" / migration.name
    path.mkdir(parents=True)
    (path / "migration.sql").write_text(migration.script)
