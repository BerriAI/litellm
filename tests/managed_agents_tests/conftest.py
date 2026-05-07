"""Shared fixtures for managed-agents integration tests.

These tests target the REAL FastAPI router stack composed in
``litellm/proxy/managed_agents_endpoints/router.py`` against a REAL local opencode HTTP
server. They exercise:

  - ``POST /v2/agents`` (real handler + a fake in-memory Prisma table)
  - ``GET /v2/sessions/:id`` (real handler against a hand-INSERTed row)
  - ``POST /v2/sessions/:id/messages`` (real handler + adapter HTTP call)
  - ``GET /v2/sessions/:id/messages`` (real handler + adapter HTTP call)
  - ``GET /v2/sessions/:id/events`` (real handler + adapter SSE stream)

The "DB" is a tiny in-memory stand-in (``_FakeDB``) that implements only
the methods the v2 handlers actually call (``find_first`` / ``create``).
This keeps the test harness portable — no Postgres, no SQLite, no Prisma
codegen — while still letting the handler chain run end-to-end against
a real opencode process.

The opencode binary is required: tests that depend on it auto-skip when
``opencode`` is missing on PATH (see ``opencode_server`` fixture).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import types
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

import httpx
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``integration`` marker locally so tests in this
    directory can be selected via ``pytest -m integration`` without
    polluting the global ``pyproject.toml`` markers list.
    """
    config.addinivalue_line(
        "markers",
        "integration: end-to-end test requiring real external processes "
        "(e.g. opencode serve)",
    )


# pytestmark applies to every test that imports this conftest's fixtures —
# tests are integration tests against a real opencode process and are not
# safe to parallelise on the same port.
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers: free-port + binary discovery
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Return a free TCP port on 127.0.0.1.

    Race conditions are possible (the kernel may hand the port to another
    process between this call and ``opencode serve --port`` binding to
    it), but the window is small and acceptable for tests.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _opencode_on_path() -> Optional[str]:
    """Return the resolved opencode binary path, or None if missing."""
    from shutil import which

    return which("opencode")


# ---------------------------------------------------------------------------
# Fake in-memory Prisma stand-in
# ---------------------------------------------------------------------------


class _FakeTable:
    """Stand-in for ``prisma_client.db.<table>``.

    Implements ``create`` / ``find_first`` / ``find_unique`` — the only
    methods the v2 managed-agents handlers call. Rows are stored as plain
    dicts; ``model_dump()`` on the returned object returns a copy so the
    handler's ``_row_to_dict`` helper produces JSON-safe dicts.
    """

    def __init__(self) -> None:
        self.rows: Dict[str, Dict[str, Any]] = {}

    async def create(self, *, data: Dict[str, Any]) -> types.SimpleNamespace:
        config = data.get("config")
        if config is not None and hasattr(config, "data"):
            stored_config: Any = config.data
        else:
            stored_config = config

        # Same handling for any other ``prisma.Json``-wrapped fields
        normalized = dict(data)
        normalized["config"] = stored_config
        for key in ("repos", "env_vars", "sandbox_metadata"):
            value = normalized.get(key)
            if value is not None and hasattr(value, "data"):
                normalized[key] = value.data

        self.rows[data["id"]] = normalized
        return types.SimpleNamespace(model_dump=lambda r=normalized: dict(r))

    async def find_first(
        self, *, where: Dict[str, Any]
    ) -> Optional[types.SimpleNamespace]:
        for row in self.rows.values():
            if all(row.get(k) == v for k, v in where.items()):
                return types.SimpleNamespace(model_dump=lambda r=row: dict(r))
        return None

    async def find_unique(
        self, *, where: Dict[str, Any]
    ) -> Optional[types.SimpleNamespace]:
        return await self.find_first(where=where)


class _FakeDB:
    """Top-level fake DB exposing the two tables the v2 handlers use."""

    def __init__(self) -> None:
        self.litellm_managedagent = _FakeTable()
        self.litellm_managedagentsession = _FakeTable()


class _FakePrismaClient:
    """Fake prisma_client carrying a ``.db`` attribute."""

    def __init__(self) -> None:
        self.db = _FakeDB()


# ---------------------------------------------------------------------------
# opencode lifecycle
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def opencode_server() -> Iterator[str]:
    """Start a real opencode server on a free port.

    Yields the base URL, e.g. ``http://127.0.0.1:54321``. Skips the test
    cleanly if ``opencode`` is not installed on PATH.

    The server is shared across the test session — start cost is ~2s and
    we don't want to repeat it. Cleanup terminates the process via SIGTERM
    with a short grace period before SIGKILL.
    """
    binary = _opencode_on_path()
    if binary is None:
        pytest.skip("opencode binary not on PATH; integration tests skipped")

    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    proc = subprocess.Popen(
        [binary, "serve", "--port", str(port), "--hostname", "127.0.0.1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ},
    )

    # Wait for /global/health to come up; bail out if it doesn't within
    # the grace window so a missing dependency fails clearly instead of
    # hanging the entire suite.
    deadline = time.time() + 30.0
    health_url = f"{base_url}/global/health"
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        if proc.poll() is not None:
            stderr = (
                proc.stderr.read().decode("utf-8", errors="replace")
                if proc.stderr
                else ""
            )
            pytest.skip(f"opencode exited early (rc={proc.returncode}): {stderr[:500]}")
        try:
            resp = httpx.get(health_url, timeout=2.0)
            if resp.status_code == 200 and resp.json().get("healthy") is True:
                break
        except (httpx.ConnectError, httpx.TimeoutException, ValueError) as e:
            last_err = e
        time.sleep(0.25)
    else:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        pytest.skip(
            f"opencode did not become healthy within 30s " f"(last err: {last_err!r})"
        )

    try:
        yield base_url
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


@pytest.fixture
def opencode_session(opencode_server: str) -> Iterator[Tuple[str, str]]:
    """Create a real opencode session via ``POST /session``.

    Yields ``(opencode_session_id, sandbox_url)``. Best-effort delete on
    teardown — we ignore errors because the session may have been killed
    by the per-test sandbox-death simulation in flow 3.
    """
    resp = httpx.post(
        f"{opencode_server}/session",
        json={},
        timeout=10.0,
    )
    if resp.status_code not in (200, 201):
        pytest.fail(
            f"opencode POST /session failed: {resp.status_code} {resp.text[:300]}"
        )
    payload = resp.json()
    oc_sid = payload.get("id")
    if not oc_sid or not isinstance(oc_sid, str):
        pytest.fail(f"opencode POST /session returned unexpected body: {payload!r}")

    try:
        yield oc_sid, opencode_server
    finally:
        try:
            httpx.delete(
                f"{opencode_server}/session/{oc_sid}",
                timeout=5.0,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Fake DB row + FastAPI app stitching
# ---------------------------------------------------------------------------


@pytest.fixture
def prisma_client_test(monkeypatch: pytest.MonkeyPatch) -> _FakePrismaClient:
    """Install a fake prisma_client onto ``litellm.proxy.proxy_server``.

    Also stubs the top-level ``prisma`` module so ``db.insert_agent`` can
    call ``prisma.Json(...)`` without a real Prisma codegen. Mirrors the
    pattern in ``tests/test_litellm/managed_agents/test_agents.py``.
    """
    import litellm.proxy.proxy_server as ps

    fake_client = _FakePrismaClient()
    monkeypatch.setattr(ps, "prisma_client", fake_client)

    fake_prisma_module = types.ModuleType("prisma")

    class _Json:
        def __init__(self, data: Any) -> None:
            self.data = data

    fake_prisma_module.Json = _Json  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prisma", fake_prisma_module)

    return fake_client


@pytest.fixture
def fake_db_session(
    opencode_session: Tuple[str, str],
    prisma_client_test: _FakePrismaClient,
) -> str:
    """Hand-INSERT a ``LiteLLM_ManagedAgentSession`` row pointing at the
    real local opencode server.

    Returns the session id (``ses_*``). Mirrors the manual psql INSERT
    documented in the v2 contract §1 / step 1.2 — bypasses Krrish's
    ``POST /v2/sessions`` endpoint, which is out of scope for v2 MVP
    integration testing.
    """
    oc_sid, sandbox_url = opencode_session

    session_id = f"ses_{uuid.uuid4().hex}"
    now = datetime.now(timezone.utc)

    prisma_client_test.db.litellm_managedagentsession.rows[session_id] = {
        "id": session_id,
        "agent_id": "agt_test_agent",
        "sandbox_type": "opencode",
        "sandbox_size": "small",
        "sandbox_timeout_minutes": 60,
        "sandbox_idle_timeout_minutes": 10,
        "sandbox_image": None,
        "sandbox_url": sandbox_url,
        "sandbox_metadata": {"opencode_session_id": oc_sid},
        "status": "ready",
        "repos": [],
        "env_vars": {},
        "created_by": "test_user",
        "created_at": now,
        "updated_at": now,
        "terminated_at": None,
    }

    return session_id


class _ProxyClient:
    """Thin wrapper around an httpx.Client so tests get a stable surface
    regardless of whether we're running in-process via TestClient or via
    a real uvicorn server.

    The methods mirror the bits of ``starlette.testclient.TestClient`` we
    actually use: ``get`` / ``post`` / ``stream``.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=30.0)

    def get(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> httpx.Response:
        return self._client.get(path, params=params, headers=headers)

    def post(
        self, path: str, *, json: Any = None, headers: Optional[Dict[str, str]] = None
    ) -> httpx.Response:
        return self._client.post(path, json=json, headers=headers)

    def stream(
        self,
        method: str,
        path: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        read_timeout: float = 5.0,
    ) -> Any:
        """Open a streaming HTTP request. ``read_timeout`` controls how
        long ``iter_lines()`` will wait between bytes before raising.

        Used for the SSE ``/v2/sessions/:id/events`` endpoint — we need
        a finite read timeout so an idle SSE channel doesn't pin the
        test forever.
        """
        timeout = httpx.Timeout(connect=5.0, read=read_timeout, write=5.0, pool=5.0)
        return self._client.stream(method, path, headers=headers, timeout=timeout)

    def close(self) -> None:
        self._client.close()


def _run_uvicorn_in_thread(
    app: Any, host: str, port: int
) -> Tuple[Any, threading.Thread]:
    """Spin up uvicorn in a background thread.

    Returns ``(server, thread)``. The caller is responsible for shutting
    the server down via ``server.should_exit = True`` and joining the
    thread.

    We use uvicorn (not starlette TestClient) because TestClient runs
    the ASGI app on the calling thread and blocks ``iter_lines()`` on
    SSE streams without a usable read timeout — a real socket-level
    server is what we need for honest streaming behaviour.
    """
    import uvicorn

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="off",  # we don't run startup/shutdown hooks in tests
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, name="uvicorn-test", daemon=True)
    thread.start()
    return server, thread


@pytest.fixture
def app_client(
    prisma_client_test: _FakePrismaClient,
) -> Iterator[_ProxyClient]:
    """Real uvicorn server hosting the v2 managed-agents router stack.

    Uses uvicorn instead of starlette ``TestClient`` so SSE streaming
    works correctly with finite read timeouts. The server is bound to a
    free port on 127.0.0.1; the test interacts via a thin ``httpx.Client``
    wrapper that exposes the small surface our tests need.
    """
    from fastapi import FastAPI

    from litellm.proxy.managed_agents_endpoints.agents import router as agents_router
    from litellm.proxy.managed_agents_endpoints.events import router as events_router
    from litellm.proxy.managed_agents_endpoints.messages import (
        router as messages_router,
    )
    from litellm.proxy.managed_agents_endpoints.sessions import (
        router as sessions_router,
    )
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    app = FastAPI()
    app.include_router(agents_router)
    app.include_router(sessions_router)
    app.include_router(messages_router)
    app.include_router(events_router)

    fake_user = UserAPIKeyAuth(
        user_id="test_user",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    app.dependency_overrides[user_api_key_auth] = lambda: fake_user

    host = "127.0.0.1"
    port = _find_free_port()
    server, thread = _run_uvicorn_in_thread(app, host, port)

    # Wait for the server to start serving — uvicorn flips
    # ``server.started`` after binding.
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if getattr(server, "started", False):
            break
        time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        pytest.fail("uvicorn test server failed to start within 10s")

    client = _ProxyClient(f"http://{host}:{port}")
    try:
        yield client
    finally:
        client.close()
        server.should_exit = True
        thread.join(timeout=10)
        app.dependency_overrides.clear()
