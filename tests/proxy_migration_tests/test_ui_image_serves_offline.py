"""Image-level regression net for arbitrary-uid boot of the UI image.

OpenShift ``restricted-v2`` ignores the image ``USER`` and assigns an
arbitrary uid in GID 0. The stock nginx base expects to start as root, so
its cache (``/var/cache/nginx``) and pid (``/run``) paths are root-owned
755 and the master process dies at startup with
``mkdir() "/var/cache/nginx/client_temp" failed (13: Permission denied)``.
The fix anchors everything nginx writes under ``/tmp`` in ``ui/nginx.conf``.

Booting the image the way that deployment does, with a read-only root
filesystem and ``/tmp`` as the only writable mount, is what catches the
whole class: a boot as the default (root) uid passes even on the broken
config.

Gated on LITELLM_IMAGE so it is skipped in the normal unit-test run and
exercised only where an image has been built (the image-scan workflow).
Requires a working docker CLI.
"""

import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator

import pytest

IMAGE = os.getenv("LITELLM_IMAGE")
CURL_IMAGE = os.getenv("LITELLM_TEST_CURL_IMAGE", "curlimages/curl:8.11.1")
UI_PORT = os.getenv("LITELLM_UI_PORT", "3000")
ARBITRARY_UID = "1001200000:0"
STARTUP_TIMEOUT_SECONDS = int(os.getenv("LITELLM_UI_STARTUP_TIMEOUT", "60"))

pytestmark = [
    pytest.mark.skipif(IMAGE is None, reason="requires a built image (set LITELLM_IMAGE)"),
    pytest.mark.skipif(shutil.which("docker") is None, reason="requires the docker CLI"),
]


def _docker(*args: str, check: bool = True) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)


@pytest.fixture()
def ui_container() -> Iterator[tuple[str, str]]:
    """The UI container as an arbitrary uid in GID 0 on a network with no egress.

    ``--read-only`` with a tmpfs on ``/tmp`` mirrors the strictest supported
    deployment: ``readOnlyRootFilesystem: true`` with an emptyDir on ``/tmp``.
    A config that writes anywhere else fails here exactly like it does on
    OpenShift.
    """
    run_id = f"uiserve-{uuid.uuid4().hex[:8]}"
    network = f"{run_id}-net"
    container = f"{run_id}-ui"

    _docker("pull", "--quiet", CURL_IMAGE)
    _docker("network", "create", "--internal", network)
    try:
        assert IMAGE is not None
        _docker(
            "run", "-d", "--name", container, "--network", network,
            "--user", ARBITRARY_UID,
            "--read-only", "--tmpfs", "/tmp",
            IMAGE,
        )
        yield network, container
    finally:
        _docker("logs", container, check=False)
        _docker("rm", "-f", container, check=False)
        _docker("network", "rm", network, check=False)


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


def _probe(network: str, container: str, path: str) -> "subprocess.CompletedProcess[str]":
    return _docker(
        "run", "--rm", "--network", network, CURL_IMAGE,
        "--silent", "--show-error", "--max-time", "10",
        "--output", "/dev/null", "--write-out", "%{http_code}",
        f"http://{container}:{UI_PORT}{path}",
        check=False,
    )


def test_ui_serves_as_arbitrary_uid_read_only(ui_container: tuple[str, str]) -> None:
    """nginx boots and serves as an arbitrary uid with a read-only root fs.

    On the pre-fix config nginx exits during startup with
    ``mkdir() "/var/cache/nginx/client_temp" failed (13: Permission denied)``
    and the running-check below fails; it never reaches the probes.
    """
    network, container = ui_container

    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    healthz = None
    while time.time() < deadline:
        if not _is_running(container):
            pytest.fail(
                f"the UI container exited during startup as uid {ARBITRARY_UID} with a "
                f"read-only root filesystem. nginx writes outside /tmp.\n"
                f"{_container_logs(container)}"
            )
        healthz = _probe(network, container, "/healthz")
        if healthz.returncode == 0 and healthz.stdout.strip() == "200":
            break
        time.sleep(2)

    assert healthz is not None and healthz.stdout.strip() == "200", (
        f"/healthz never answered 200 within {STARTUP_TIMEOUT_SECONDS}s as uid "
        f"{ARBITRARY_UID}.\n{_container_logs(container)}"
    )

    for path in ("/", "/ui", "/ui/login"):
        page = _probe(network, container, path)
        assert page.stdout.strip() == "200", (
            f"GET {path} returned {page.stdout.strip()!r} as uid {ARBITRARY_UID}.\n"
            f"{_container_logs(container)}"
        )
