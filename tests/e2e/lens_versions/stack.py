from __future__ import annotations

import os
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from uuid import uuid4

from e2e_http import Headers, NoBody, ProbeResult
from e2e_metadata import step
from pydantic import Field, TypeAdapter
from transport import HttpTransport

REPOSITORY: Final = Path(__file__).resolve().parents[3]
COMPOSE: Final = Path(__file__).with_name("compose.yaml")
MASTER_KEY: Final = "sk-lens-versions-fixture"
UI_USERNAME: Final = "admin"
UI_PASSWORD: Final = "lens-versions-fixture"
MODEL: Final = "lens-versions-model"


class SessionHeaders(Headers):
    cookie: str = Field(repr=False)


@step("Run a docker command")
def docker(*args: str, env: dict[str, str] | None = None, timeout: float = 300) -> str:
    result: Final = subprocess.run(
        ("docker", *args), capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})}
    )
    assert result.returncode == 0, f"Docker operation failed: {result.stderr}"
    return result.stdout.strip()


@step("Wait for {description}")
def until(description: str, condition: Callable[[], bool], seconds: float = 240) -> None:
    deadline: Final = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.5)
    raise AssertionError(f"Timed out waiting for {description}")


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return TypeAdapter(tuple[str, int]).validate_python(listener.getsockname())[1]


def setting(directory: Path, name: str) -> str:
    return next(
        line.removeprefix(f"{name}=")
        for line in (directory / ".env").read_text().splitlines()
        if line.startswith(f"{name}=")
    )


@dataclass(frozen=True, slots=True)
class Stack:
    """A gateway from this checkout connected to one Lens release, each with its own storage"""

    gateway: HttpTransport
    lens: HttpTransport = field(repr=False)

    @step("Read {path} on the gateway as the signed in dashboard user")
    def page(self, path: str, session: str) -> ProbeResult:
        return self.gateway.probe(path, params=NoBody(), headers=SessionHeaders(cookie=f"token={session}"))

    @step("Read trace {trace_id} from Lens directly")
    def stored_trace(self, trace_id: str) -> ProbeResult:
        return self.lens.probe(f"/v1/traces/{trace_id}", params=NoBody())


@step("Start Lens from {image} with its own ClickHouse")
@contextmanager
def lens(image: str, directory: Path) -> Generator[tuple[HttpTransport, dict[str, str]]]:
    project: Final = f"lens-versions-{uuid4().hex[:12]}"
    port: Final = free_port()
    shared: Final = {
        "LENS_GATEWAY_SECRET": secrets.token_urlsafe(32),
        "LITELLM_LENS_SERVICE_TOKEN": secrets.token_urlsafe(32),
    }
    directory.mkdir(parents=True, exist_ok=True)
    compose: Final = ("compose", "-p", project, "--project-directory", str(directory), "-f", str(COMPOSE))
    env: Final = {"LENS_IMAGE": image, "LENS_PORT": str(port), "LENS_PUBLIC_URL": f"http://127.0.0.1:{port}"}
    try:
        docker(*compose, "run", "--rm", "credentials", env=env, timeout=600)
        with (directory / ".env").open("a") as generated:
            generated.write("".join(f"{name}={value}\n" for name, value in shared.items()))
        docker(*compose, "up", "-d", "--wait", env=env, timeout=600)
        yield (
            HttpTransport(f"http://127.0.0.1:{port}", setting(directory, "LENS_ADMIN_TOKEN"), 15),
            {**shared, "LITELLM_LENS_URL": f"http://127.0.0.1:{port}"},
        )
    finally:
        logs: Final = subprocess.run(
            ("docker", *compose, "logs", "lens"), capture_output=True, text=True, timeout=60, env={**os.environ, **env}
        )
        (directory / "lens.log").write_text(logs.stdout + logs.stderr)
        subprocess.run(
            ("docker", *compose, "down", "-v", "--remove-orphans"),
            capture_output=True,
            timeout=180,
            env={**os.environ, **env},
            check=True,
        )


@step("Start a PostgreSQL container for the gateway")
@contextmanager
def postgres() -> Generator[str]:
    name: Final = f"lens-versions-db-{uuid4().hex[:12]}"
    try:
        docker(
            "run", "-d", "--name", name, "-e", "POSTGRES_PASSWORD=lens", "-e", "POSTGRES_DB=litellm",
            "-p", "127.0.0.1::5432", "postgres:16",
        )  # fmt: skip
        port: Final = int(docker("port", name, "5432/tcp").rsplit(":", 1)[1])
        until(
            "PostgreSQL to accept connections",
            lambda: (
                subprocess.run(
                    ("docker", "exec", name, "pg_isready", "-U", "postgres"), capture_output=True, timeout=30
                ).returncode
                == 0
            ),
        )
        yield f"postgresql://postgres:lens@127.0.0.1:{port}/litellm"
    finally:
        subprocess.run(("docker", "rm", "-f", "-v", name), capture_output=True, timeout=60, check=True)


@step("Start the gateway from this checkout, connected to Lens")
@contextmanager
def gateway(database: str, connection: dict[str, str], directory: Path) -> Generator[HttpTransport]:
    port: Final = free_port()
    (directory / "config.yaml").write_text(
        f"model_list:\n  - model_name: {MODEL}\n    litellm_params:\n      model: openai/{MODEL}\n"
        "      api_key: none\n      mock_response: served by the Lens versions fixture\n"
        "general_settings:\n  master_key: os.environ/LITELLM_MASTER_KEY\n"
    )
    with (directory / "gateway.log").open("w") as log:
        process: Final = subprocess.Popen(  # test-quality-ok: the gateway under test is this checkout
            (
                sys.executable,
                "litellm/proxy/proxy_cli.py",
                "--config",
                str(directory / "config.yaml"),
                "--port",
                str(port),
                "--use_v2_migration_resolver",
            ),  # fmt: skip
            cwd=REPOSITORY,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                **connection,
                "DATABASE_URL": database,
                "LITELLM_MASTER_KEY": MASTER_KEY,
                "LITELLM_SALT_KEY": MASTER_KEY,
                "UI_USERNAME": UI_USERNAME,
                "UI_PASSWORD": UI_PASSWORD,
                "LITELLM_TELEMETRY": "False",
            },
        )
        transport: Final = HttpTransport(f"http://127.0.0.1:{port}", MASTER_KEY, 30)

        def ready() -> bool:
            assert process.poll() is None, f"The gateway exited during startup, see {directory / 'gateway.log'}"
            return transport.probe("/health/readiness", params=NoBody()).status_code == 200

        try:
            until("the gateway to be ready", ready, seconds=420)
            yield transport
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()


@contextmanager
def stack(image: str, directory: Path) -> Generator[Stack]:
    with lens(image, directory) as (lens_transport, connection), postgres() as database:
        with gateway(database, connection, directory) as gateway_transport:
            yield Stack(gateway=gateway_transport, lens=lens_transport)
