"""Image-level regression net for the prisma bake in the componentized images.

The gateway and backend serve requests; they never shell out to the Prisma CLI
(``PrismaManager.setup_database`` is reachable only from ``proxy_cli.py``, which
uvicorn'ing ``gateway.main:app`` bypasses). What they do need is the generated
client's baked query engine, and prisma-python resolves those baked paths
eagerly, with an existence check that propagates EACCES rather than skipping the
candidate. An engine baked under a build-time ``HOME`` is therefore unreadable to
any other runtime uid, and the process dies during startup before
``PRISMA_QUERY_ENGINE_BINARY`` is ever consulted.

That is what an OpenShift ``restricted-v2`` namespace produces: the image
``USER`` is ignored and an arbitrary uid in GID 0 is assigned instead. The
symptom is not a degraded proxy, it is a proxy that does not serve at all.

Booting the image the way that deployment does, and requiring it to answer a
request with a live database connection, is what catches the whole class:
a boot as the default uid, or one that reaches the internet, passes even when
the bake is unusable everywhere it actually ships.

Gated on LITELLM_IMAGE (the tag of the image to exercise) so it is skipped in
the normal unit-test run and exercised only where an image has been built (the
image-scan workflow). Requires a working docker CLI.
"""

import json
import shutil
import subprocess
import time
import uuid

import os
import pytest

IMAGE = os.getenv("LITELLM_IMAGE")
POSTGRES_IMAGE = os.getenv("LITELLM_TEST_POSTGRES_IMAGE", "postgres:16-alpine")
CURL_IMAGE = os.getenv("LITELLM_TEST_CURL_IMAGE", "curlimages/curl:8.11.1")
COMPONENT_PORT = os.getenv("LITELLM_COMPONENT_PORT", "4000")
NON_ROOT_UID = "12345:0"
STARTUP_TIMEOUT_SECONDS = int(os.getenv("LITELLM_COMPONENT_STARTUP_TIMEOUT", "180"))

pytestmark = [
    pytest.mark.skipif(IMAGE is None, reason="requires a built image (set LITELLM_IMAGE)"),
    pytest.mark.skipif(shutil.which("docker") is None, reason="requires the docker CLI"),
]


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=check
    )


@pytest.fixture()
def offline_stack():
    """A component container and a fresh Postgres on a network with no egress.

    NON_ROOT_UID is an arbitrary uid in GID 0, the shape OpenShift restricted-v2
    assigns. Postgres and curl are pulled while egress still exists, because the
    ``--internal`` network below has none: that is what makes a prisma engine
    download (binaries.prisma.sh / npm) fail rather than mask a bake that is not
    self-contained.

    The container runs with DISABLE_SCHEMA_UPDATE, since applying the schema is
    the migration job's responsibility in this topology and needs the Prisma CLI
    these images deliberately omit, and with LITELLM_LOCAL_MODEL_COST_MAP, or the
    proxy spends the whole startup budget timing out on a cost-map fetch over the
    network it does not have.

    Yields (network_name, component_container). Both are torn down afterwards.
    """
    run_id = f"componentserve-{uuid.uuid4().hex[:8]}"
    network = f"{run_id}-net"
    pg = f"{run_id}-pg"
    component = f"{run_id}-app"

    _docker("pull", "--quiet", POSTGRES_IMAGE)
    _docker("pull", "--quiet", CURL_IMAGE)
    _docker("network", "create", "--internal", network)
    try:
        _docker(
            "run", "-d", "--name", pg, "--network", network,
            "-e", "POSTGRES_PASSWORD=pw", "-e", "POSTGRES_DB=litellm",
            POSTGRES_IMAGE,
        )
        _wait_until_postgres_ready(pg)
        assert IMAGE is not None
        _docker(
            "run", "-d", "--name", component, "--network", network,
            "--user", NON_ROOT_UID,
            "-e", f"DATABASE_URL=postgresql://postgres:pw@{pg}:5432/litellm",
            "-e", "LITELLM_MASTER_KEY=sk-component-serve-test",
            "-e", "DISABLE_SCHEMA_UPDATE=true",
            "-e", "LITELLM_LOCAL_MODEL_COST_MAP=True",
            IMAGE,
        )
        yield network, component
    finally:
        _docker("logs", component, check=False)
        _docker("rm", "-f", component, check=False)
        _docker("rm", "-f", pg, check=False)
        _docker("network", "rm", network, check=False)


def _wait_until_postgres_ready(pg: str, attempts: int = 60) -> None:
    for _ in range(attempts):
        running = _docker(
            "ps", "--filter", f"name={pg}", "--filter", "status=running",
            "--format", "{{.Names}}", check=False,
        ).stdout
        if pg not in running:
            logs = _docker("logs", pg, check=False)
            pytest.fail(f"postgres container is not running:\n{logs.stdout}\n{logs.stderr}")
        ready = _docker(
            "exec", pg, "pg_isready", "-U", "postgres", "-d", "litellm", check=False
        )
        if ready.returncode == 0:
            return
        time.sleep(1)
    pytest.fail(f"postgres never became ready after {attempts}s")


def _container_logs(container: str) -> str:
    logs = _docker("logs", container, check=False)
    return f"stdout:\n{logs.stdout}\nstderr:\n{logs.stderr}"


def _is_running(container: str) -> bool:
    return bool(
        _docker(
            "ps", "--filter", f"name={container}", "--filter", "status=running",
            "--format", "{{.Names}}", check=False,
        ).stdout.strip()
    )


def _readiness(network: str, component: str) -> subprocess.CompletedProcess:
    """Ask the component for its readiness, from a peer on the same egress-less network."""
    return _docker(
        "run", "--rm", "--network", network, CURL_IMAGE,
        "--silent", "--max-time", "10",
        f"http://{component}:{COMPONENT_PORT}/health/readiness",
        check=False,
    )


def test_component_serves_offline_as_non_root_uid(offline_stack):
    """The component answers a request with a live DB connection, offline, as an arbitrary uid.

    On the pre-fix image this never gets a response: the engine baked under
    /home/nonroot (mode 0700, owned by uid 65532) raises
    ``PermissionError: .../query-engine-linux-...`` out of pathlib and uvicorn
    reports ``Application startup failed. Exiting.``. A bake at the fixed,
    world-readable /opt/prisma is what lets any uid start the client.

    `db: connected` is the load-bearing part of the assertion: it means the
    query engine binary was found, executed, and reached Postgres. A liveness
    probe alone would pass on an image whose engine never resolved.
    """
    network, component = offline_stack

    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    probe = None
    while time.time() < deadline:
        if not _is_running(component):
            pytest.fail(
                f"the component exited during startup as uid {NON_ROOT_UID} with no egress. "
                "The prisma bake is not readable to a uid other than the one that built it, "
                "so the proxy does not serve at all.\n"
                f"{_container_logs(component)}"
            )
        probe = _readiness(network, component)
        if probe.returncode == 0 and probe.stdout.strip():
            break
        time.sleep(2)

    assert probe is not None and probe.returncode == 0 and probe.stdout.strip(), (
        f"/health/readiness never answered within {STARTUP_TIMEOUT_SECONDS}s as uid "
        f"{NON_ROOT_UID} with no egress.\n{_container_logs(component)}"
    )

    payload = json.loads(probe.stdout)
    assert payload.get("db") == "connected", (
        f"the component answered but its database is {payload.get('db')!r}, so the baked "
        f"query engine did not resolve as uid {NON_ROOT_UID}.\nresponse: {probe.stdout}\n"
        f"{_container_logs(component)}"
    )
