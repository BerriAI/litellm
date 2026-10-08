"""An owned, source-built proxy against a real RDS writer and cross-region replica.

Only this child process carries the IAM env, so the shared e2e stack stays
untouched. The proxy's own AWS_REGION is a third region, distinct from both the
writer's and the replica's, so a boot that succeeds proves each connection was
signed in its own region rather than the ambient one.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol, cast

import boto3
import psycopg
from e2e_config import INHERITED_ENV_PREFIXES, MASTER_KEY, available_port
from e2e_http import NoBody
from idp import stop_process_group
from proxy_client import ProxyClient, build_proxy_client
from psycopg.rows import class_row
from pydantic import BaseModel

RDS_HOSTNAME_PATTERN: Final = re.compile(r"\.([a-z0-9-]+)\.rds\.amazonaws\.com")
THIRD_REGION_CANDIDATES: Final = ("us-west-2", "eu-west-1", "ap-southeast-2")
NOVA_MICRO_MODEL: Final = "e2e-rds-nova-micro"


class ReplicaConnectionRow(BaseModel):
    pid: int
    state: str


def hostname_region(hostname: str) -> str:
    match: Final = RDS_HOSTNAME_PATTERN.search(hostname)
    assert match, f"{hostname} is not an RDS hostname"
    return match.group(1)


class RdsTokenClient(Protocol):
    def generate_db_auth_token(self, DBHostname: str, Port: int, DBUsername: str) -> str: ...


def replica_connections(reader_host: str, reader_region: str, user: str, database: str) -> list[ReplicaConnectionRow]:
    client: Final = cast(RdsTokenClient, boto3.client("rds", region_name=reader_region))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # rds has no installed stub, so the boto3 overload returns Unknown
    token: Final = client.generate_db_auth_token(reader_host, 5432, user)
    with psycopg.Connection[ReplicaConnectionRow].connect(
        f"host={reader_host} port=5432 user={user} password={token} dbname={database} "
        "sslmode=require connect_timeout=15",
        row_factory=class_row(ReplicaConnectionRow),
    ) as conn:
        rows: Final = conn.execute(
            "SELECT pid, state FROM pg_stat_activity WHERE usename = %s AND pid <> pg_backend_pid()",
            (user,),
        ).fetchall()
    return list(rows)


@dataclass(slots=True)
class RdsGateway:
    base_url: str
    proxy: ProxyClient
    log_path: Path
    writer_region: str
    reader_region: str
    _environment: Mapping[str, str] = field(repr=False)
    _command: tuple[str, ...] = field(repr=False)
    _child: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        with self.log_path.open("ab") as log:
            self._child = subprocess.Popen(
                self._command,
                env=self._environment,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        deadline: Final = time.monotonic() + 240
        while time.monotonic() < deadline:
            assert self._child.poll() is None, f"owned RDS gateway exited; inspect {self.log_path}"
            result = self.proxy.transport.probe("/health/liveliness", params=NoBody())
            if result.status_code == 200:
                return
            time.sleep(0.5)
        raise AssertionError(f"owned RDS gateway did not become ready; inspect {self.log_path}")

    def stop(self) -> None:
        if self._child is not None:
            stop_process_group(self._child)
            assert self._child.poll() is not None, "RDS gateway process is still alive"

    def log_text(self) -> str:
        return self.log_path.read_text() if self.log_path.exists() else ""


def owned_rds_gateway(directory: Path, cleanup: ExitStack, overrides: Mapping[str, str]) -> RdsGateway:
    inputs: Final = ("E2E_RDS_WRITER_HOST", "E2E_RDS_READER_HOST", "E2E_RDS_USER", "E2E_RDS_DATABASE")
    for name in inputs:
        assert os.environ.get(name), f"{name} is required for the owned RDS gateway"
    writer_host: Final = os.environ["E2E_RDS_WRITER_HOST"]
    reader_host: Final = os.environ["E2E_RDS_READER_HOST"]
    user: Final = os.environ["E2E_RDS_USER"]
    database: Final = os.environ["E2E_RDS_DATABASE"]
    writer_region: Final = hostname_region(writer_host)
    reader_region: Final = hostname_region(reader_host)
    assert writer_region != reader_region, (
        f"writer {writer_host} and replica {reader_host} must sit in different regions"
    )
    proxy_region: Final = next(
        region for region in THIRD_REGION_CANDIDATES if region not in (writer_region, reader_region)
    )
    port: Final = available_port()
    base_url: Final = f"http://127.0.0.1:{port}"

    config: Final = directory / "rds-gateway.yaml"
    config.write_text(
        "model_list:\n"
        f"  - model_name: {NOVA_MICRO_MODEL}\n"
        "    litellm_params:\n"
        "      model: bedrock/us.amazon.nova-micro-v1:0\n"
        "general_settings:\n"
        "  master_key: os.environ/LITELLM_MASTER_KEY\n"
    )
    stripped: Final = frozenset(
        (
            "DATABASE_URL",
            "DIRECT_URL",
            "AWS_RDS_REGION",
            "AWS_RDS_READ_REPLICA_REGION",
            "AWS_REGION",
            "AWS_REGION_NAME",
        )
    )
    environment: Final = {
        **{
            key: value
            for key, value in os.environ.items()
            if not key.startswith(INHERITED_ENV_PREFIXES) and key not in stripped
        },
        "AWS_REGION": proxy_region,
        "IAM_TOKEN_DB_AUTH": "true",
        "DATABASE_HOST": writer_host,
        "DATABASE_PORT": "5432",
        "DATABASE_USER": user,
        "DATABASE_NAME": database,
        "DATABASE_URL_READ_REPLICA": f"postgresql://{user}@{reader_host}:5432/{database}?sslmode=require",
        "LITELLM_MASTER_KEY": MASTER_KEY,
        "LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY": "true",
        "DISABLE_SCHEMA_UPDATE": "true",
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        **overrides,
    }
    gateway: Final = RdsGateway(
        base_url=base_url,
        proxy=build_proxy_client(
            base_url=base_url,
            control_plane_base_url=base_url,
            replica_urls=(base_url,),
            control_replica_urls=(base_url,),
            master_key=MASTER_KEY,
        ),
        log_path=directory / "rds-gateway.log",
        writer_region=writer_region,
        reader_region=reader_region,
        _environment=environment,
        _command=(sys.executable, "-m", "litellm.proxy.proxy_cli", "--config", str(config), "--port", str(port)),
    )
    cleanup.callback(gateway.stop)
    gateway.start()
    return gateway
