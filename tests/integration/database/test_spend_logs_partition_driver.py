import os
import socket
import subprocess
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import psycopg
from integration._support.client import Gateway
from integration._support.process import owned_proxy_process
from packaging.requirements import Requirement
from psycopg import sql

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
PRISMA_DIR: Final = REPO_ROOT / "litellm-proxy-extras" / "litellm_proxy_extras"
PARTITION_SCRIPT: Final = REPO_ROOT / "db_scripts" / "partition_spend_logs.sql"
IMAGE_DOCKERFILES: Final = (
    REPO_ROOT / "Dockerfile",
    REPO_ROOT / "docker" / "Dockerfile.database",
    REPO_ROOT / "docker" / "Dockerfile.non_root",
    REPO_ROOT / "backend" / "Dockerfile",
    REPO_ROOT / "gateway" / "Dockerfile",
    REPO_ROOT / "migrations" / "Dockerfile",
)
PURE_PYTHON_DRIVER: Final = {"PSYCOPG_IMPL": "python"}


def locked_extra_proxy_packages() -> frozenset[str]:
    export: Final = subprocess.run(
        ["uv", "export", "--frozen", "--no-hashes", "--no-dev", "--no-emit-project", "--extra", "extra_proxy"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )
    return frozenset(
        Requirement(line.split(" ;")[0]).name.lower().replace("_", "-")
        for line in export.stdout.splitlines()
        if line and not line.startswith(("#", "-", " "))
    )


def free_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return reserve.getsockname()[1]


def runtime_stage(dockerfile: Path) -> str:
    return dockerfile.read_text().rsplit("FROM $LITELLM_RUNTIME_IMAGE", maxsplit=1)[1]


@contextmanager
def partitioned_database() -> Iterator[str]:
    name: Final = f"integration_partitioned_{uuid.uuid4().hex}"
    admin_url: Final = os.environ["DATABASE_URL"]
    database_url: Final = urlunsplit(urlsplit(admin_url)._replace(path=f"/{name}"))
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-m",
                    "prisma",
                    "migrate",
                    "deploy",
                    "--schema",
                    str(PRISMA_DIR / "schema.prisma"),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ, "DATABASE_URL": database_url},
            )
            with psycopg.connect(database_url, autocommit=True) as connection:
                connection.execute(PARTITION_SCRIPT.read_text())
            yield database_url
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


def test_proxy_image_extra_ships_psycopg_without_the_binary_wheel_and_uses_the_image_libpq() -> None:
    packages: Final = locked_extra_proxy_packages()
    assert "psycopg" in packages, sorted(packages)
    assert "psycopg-binary" not in packages, sorted(packages)
    for dockerfile in IMAGE_DOCKERFILES:
        assert "libpq" in runtime_stage(dockerfile), f"{dockerfile.name} runtime stage installs no libpq for psycopg"


def test_pure_python_psycopg_detects_partitioned_spend_logs_and_refuses_db_push(gateway: Gateway) -> None:
    with partitioned_database() as database_url:
        proxy: Final = subprocess.run(
            [
                sys.executable,
                "-P",
                "-m",
                "integration._support.proxy",
                "--config",
                "tests/integration/proxy_config.yaml",
                "--host",
                "127.0.0.1",
                "--port",
                str(free_port()),
                "--use_prisma_db_push",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=REPO_ROOT,
            env={
                **os.environ,
                **PURE_PYTHON_DRIVER,
                "DATABASE_URL": database_url,
                "LITELLM_MASTER_KEY": gateway.key,
                "LITELLM_SALT_KEY": os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt"),
            },
        )
        output: Final = proxy.stdout + proxy.stderr
        assert proxy.returncode != 0, output
        assert "LiteLLM_SpendLogs is a partitioned table" in output, output
        assert "psycopg is not installed" not in output, output
        with psycopg.connect(database_url) as connection:
            partitioned: Final = connection.execute(
                "SELECT 1 FROM pg_partitioned_table pt JOIN pg_class c ON c.oid = pt.partrelid "
                "WHERE c.relname = 'LiteLLM_SpendLogs'"
            ).fetchone()
        assert partitioned is not None, "db push rewrote the partitioned LiteLLM_SpendLogs table"


def test_pure_python_psycopg_lets_an_unpartitioned_proxy_boot_and_serve(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(gateway, tmp_path, PURE_PYTHON_DRIVER) as proxy:
        health: Final = proxy.gateway.request("GET", "/health/readiness")
        assert health.status_code == 200, health.text
        assert health.json()["db"] == "connected", health.text
        log: Final = proxy.log.read_text()
        assert "psycopg is not installed" not in log, log
