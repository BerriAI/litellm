"""Image-level regression net for the prisma bake in the shipped runtime image.

Boots a built image's migration entrypoint the way an OpenShift / air-gapped
deployment does (an internal-only network with no egress, an arbitrary non-root
uid in GID 0) against a brand-new Postgres, and asserts the schema was created.

This catches the whole failure class, not one symptom: a bake that only works
under `docker run` as the default uid with network still passes every existing
check, because the migration entrypoint exits 0 even when it applied nothing.
Asserting the table count is what turns that silent success into a hard fail.

Gated on LITELLM_IMAGE (the tag of the image to exercise) so it is skipped in
the normal unit-test run and exercised only where an image has been built (the
image-scan workflow). Requires a working docker CLI.
"""

import shutil
import subprocess
import uuid

import os
import pytest

IMAGE = os.getenv("LITELLM_IMAGE")
POSTGRES_IMAGE = os.getenv("LITELLM_TEST_POSTGRES_IMAGE", "postgres:16-alpine")
MIN_TABLES = int(os.getenv("LITELLM_TEST_MIN_TABLES", "20"))
NON_ROOT_UID = "12345:0"  # arbitrary uid in GID 0, as OpenShift restricted-v2 assigns

pytestmark = [
    pytest.mark.skipif(IMAGE is None, reason="requires a built image (set LITELLM_IMAGE)"),
    pytest.mark.skipif(shutil.which("docker") is None, reason="requires the docker CLI"),
]


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=check
    )


@pytest.fixture()
def offline_postgres():
    """A fresh Postgres reachable only over an internal-only (no egress) network.

    Yields (network_name, postgres_host). Both are torn down afterwards.
    """
    run_id = f"offlinemig-{uuid.uuid4().hex[:8]}"
    network = f"{run_id}-net"
    pg = f"{run_id}-pg"

    # Pull Postgres while egress still exists; the internal network below has none.
    _docker("pull", "--quiet", POSTGRES_IMAGE)
    # --internal => containers on this network cannot reach the internet, so a
    # prisma engine download (binaries.prisma.sh / npm) fails instead of masking
    # a non-self-contained bake.
    _docker("network", "create", "--internal", network)
    try:
        _docker(
            "run", "-d", "--name", pg, "--network", network,
            "-e", "POSTGRES_PASSWORD=pw", "-e", "POSTGRES_DB=litellm",
            POSTGRES_IMAGE,
        )
        _wait_until_ready(pg)
        yield network, pg
    finally:
        _docker("rm", "-f", pg, check=False)
        _docker("network", "rm", network, check=False)


def _wait_until_ready(pg: str, attempts: int = 60) -> None:
    for _ in range(attempts):
        running = _docker(
            "ps", "--filter", f"name={pg}", "--filter", "status=running",
            "--format", "{{.Names}}", check=False,
        ).stdout
        if pg not in running:
            logs = _docker("logs", pg, check=False).stdout + _docker("logs", pg, check=False).stderr
            pytest.fail(f"postgres container is not running:\n{logs}")
        ready = _docker(
            "exec", pg, "pg_isready", "-U", "postgres", "-d", "litellm", check=False
        )
        if ready.returncode == 0:
            return
        subprocess.run(["sleep", "1"])
    pytest.fail(f"postgres never became ready after {attempts}s")


def _table_count(pg: str) -> int:
    result = _docker(
        "exec", pg, "psql", "-U", "postgres", "-d", "litellm", "-tAc",
        "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';",
    )
    return int(result.stdout.strip() or "0")


def test_migration_offline_as_non_root_uid(offline_postgres):
    """The migration entrypoint creates the full schema offline as an arbitrary uid.

    Reproduces the OpenShift / air-gapped failure: on the pre-fix image the
    migration exits 0 having created 0 tables (every DB endpoint then 500s on
    missing columns); a self-contained bake creates the full schema.
    """
    network, pg = offline_postgres
    assert IMAGE is not None

    migrate = _docker(
        "run", "--rm", "--network", network, "--user", NON_ROOT_UID,
        "-e", f"DATABASE_URL=postgresql://postgres:pw@{pg}:5432/litellm",
        "-e", "LITELLM_MASTER_KEY=sk-offline-migration-test",
        "-e", "DISABLE_SCHEMA_UPDATE=false",
        "-w", "/app", "--entrypoint", "python",
        IMAGE, "litellm/proxy/prisma_migration.py",
        check=False,
    )
    tables = _table_count(pg)

    assert migrate.returncode == 0, (
        f"migration entrypoint exited {migrate.returncode} offline as uid {NON_ROOT_UID}\n"
        f"stdout:\n{migrate.stdout}\nstderr:\n{migrate.stderr}"
    )
    assert tables >= MIN_TABLES, (
        f"only {tables} tables created (need >= {MIN_TABLES}) offline as uid {NON_ROOT_UID}. "
        "The prisma bake is not self-contained: it needs a runtime download or a "
        "writable HOME/cache, so OpenShift and air-gapped deployments start on an "
        f"empty database.\nstdout:\n{migrate.stdout}\nstderr:\n{migrate.stderr}"
    )


def test_runtime_cache_env_not_read_only():
    """No runtime cache env var may point at the world-read-only /opt/prisma bake.

    /opt/prisma is baked `a+rX` (no write). Pointing XDG_CACHE_HOME (or any cache
    var an XDG-aware library honours) there would deny writes for every uid, so
    guard against a future edit reintroducing that.
    """
    assert IMAGE is not None
    env = _docker("run", "--rm", "--entrypoint", "env", IMAGE).stdout
    offenders = [
        line for line in env.splitlines()
        if line.startswith(("XDG_CACHE_HOME=", "XDG_DATA_HOME=", "HOME="))
        and line.split("=", 1)[1].startswith("/opt/prisma")
    ]
    assert not offenders, f"cache/home env points at the read-only bake: {offenders}"
