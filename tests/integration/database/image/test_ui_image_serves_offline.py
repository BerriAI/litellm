"""Image-level regression net for the standalone UI image.

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

The Next.js client requests RSC payloads under the dashboard's ``/ui`` URL
prefix, while the static export stores them at the nginx document root. The
image must strip that prefix for existing payloads without serving HTML for a
missing payload.

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
        _wait_for_ui(network, container)
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


def _fetch_body(network: str, container: str, path: str) -> "subprocess.CompletedProcess[str]":
    """Fetch a payload body and make curl fail on an HTTP error response."""
    return _docker(
        "run", "--rm", "--network", network, CURL_IMAGE,
        "--silent", "--show-error", "--fail", "--max-time", "10",
        f"http://{container}:{UI_PORT}{path}",
        check=False,
    )


def _probe_content_type(network: str, container: str, path: str) -> "subprocess.CompletedProcess[str]":
    """Read the response MIME type without retaining another payload body."""
    return _docker(
        "run", "--rm", "--network", network, CURL_IMAGE,
        "--silent", "--show-error", "--fail", "--max-time", "10",
        "--output", "/dev/null", "--write-out", "%{content_type}",
        f"http://{container}:{UI_PORT}{path}",
        check=False,
    )


def _wait_for_ui(network: str, container: str) -> None:
    """Wait for nginx to serve traffic or fail with its startup logs."""
    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    while time.time() < deadline:
        if not _is_running(container):
            pytest.fail(
                f"the UI container exited during startup as uid {ARBITRARY_UID} with a "
                f"read-only root filesystem. nginx writes outside /tmp.\n"
                f"{_container_logs(container)}"
            )
        healthz = _probe(network, container, "/healthz")
        if healthz.returncode == 0 and healthz.stdout.strip() == "200":
            return
        time.sleep(2)

    pytest.fail(
        f"/healthz never answered 200 within {STARTUP_TIMEOUT_SECONDS}s as uid "
        f"{ARBITRARY_UID}.\n{_container_logs(container)}"
    )


def test_ui_serves_as_arbitrary_uid_read_only(ui_container: tuple[str, str]) -> None:
    """nginx boots and serves as an arbitrary uid with a read-only root fs.

    On the pre-fix config nginx exits during startup with
    ``mkdir() "/var/cache/nginx/client_temp" failed (13: Permission denied)``
    and the fixture fails before it reaches these route probes.
    """
    network, container = ui_container

    for path in ("/", "/ui", "/ui/login"):
        page = _probe(network, container, path)
        assert page.stdout.strip() == "200", (
            f"GET {path} returned {page.stdout.strip()!r} as uid {ARBITRARY_UID}.\n"
            f"{_container_logs(container)}"
        )


@pytest.mark.parametrize(
    ("canonical_path", "ui_path"),
    (
        ("/index.txt", "/ui/index.txt"),
        ("/api-keys/index.txt", "/ui/api-keys/index.txt"),
        ("/api-keys/index.txt", "/ui/api-keys.txt"),
        ("/__next._tree.txt", "/ui/__next._tree.txt"),
        ("/api-keys/__next._tree.txt", "/ui/api-keys/__next._tree.txt"),
    ),
    ids=("dashboard-root", "trailing-slash-route", "route", "root-tree", "nested-tree"),
)
def test_ui_prefixed_rsc_payload_matches_static_export(
    ui_container: tuple[str, str], canonical_path: str, ui_path: str,
) -> None:
    network, container = ui_container

    canonical_payload = _fetch_body(network, container, canonical_path)
    ui_payload = _fetch_body(network, container, f"{ui_path}?_rsc=regression")
    ui_content_type = _probe_content_type(network, container, f"{ui_path}?_rsc=regression")

    assert canonical_payload.returncode == 0, canonical_payload.stderr
    assert ui_payload.returncode == 0, ui_payload.stderr
    assert ui_content_type.returncode == 0, ui_content_type.stderr
    assert canonical_payload.stdout
    assert ui_payload.stdout == canonical_payload.stdout
    assert ui_content_type.stdout.strip().startswith("text/plain")


def test_ui_prefixed_missing_rsc_payload_stays_404(ui_container: tuple[str, str]) -> None:
    network, container = ui_container

    missing_payload = _probe(network, container, "/ui/does-not-exist.txt?_rsc=regression")

    assert missing_payload.stdout.strip() == "404", _container_logs(container)
