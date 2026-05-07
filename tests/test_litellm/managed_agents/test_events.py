"""Unit tests for `GET /v2/sessions/:id/events` (LIT-2920).

Covers contract §6.6 + §7:
- SSE stream format (one frame per event, blank line separator).
- Required SSE headers (`Cache-Control: no-cache`, `Connection: keep-alive`,
  `X-Accel-Buffering: no`, `Content-Type: text/event-stream`).
- Pre-forward checks happen before streaming begins:
    - 404 when the session does not exist for this caller.
    - 503 when the session is still provisioning.
    - 404 when the session is in a terminal state (terminated/error).
- Mid-stream `SandboxUnreachableError` is converted to an inline `error`
  SSE event (status code is already 200 by then).

The handler depends on:
- `user_api_key_auth` (FastAPI dep) — overridden via `app.dependency_overrides`.
- `litellm.proxy.proxy_server.prisma_client` — patched to a sentinel.
- `litellm.proxy.managed_agents_endpoints.events.get_session` — patched per test.
- `litellm.proxy.managed_agents_endpoints.events.get_adapter` — patched per test.
"""

from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.managed_agents.adapters.base import SandboxUnreachableError
from litellm.proxy.managed_agents_endpoints.events import router as events_router
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


CALLER_USER_ID = "user_xyz"


def _build_app() -> FastAPI:
    """Build a FastAPI app with the events router and auth override."""
    app = FastAPI()
    app.include_router(events_router)

    async def _fake_auth() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_id=CALLER_USER_ID, api_key="sk-test")

    app.dependency_overrides[user_api_key_auth] = _fake_auth
    return app


def _ready_session_row(
    session_id: str = "ses_test",
    sandbox_url: str = "http://127.0.0.1:1234",
    oc_sid: str = "oc_sid_abc",
) -> Dict[str, Any]:
    return {
        "id": session_id,
        "agent_id": "agt_abc",
        "sandbox_type": "opencode",
        "sandbox_url": sandbox_url,
        "sandbox_metadata": {"opencode_session_id": oc_sid},
        "status": "ready",
        "created_by": CALLER_USER_ID,
    }


class _FakeAdapter:
    """Adapter stub matching the SandboxAdapter Protocol's stream_events.

    Yields the supplied list of (event_type, data) tuples in order. If
    `raise_after` is set, raises SandboxUnreachableError after that many
    events have been yielded — used to test mid-stream error handling.
    """

    def __init__(
        self,
        events: List[Tuple[str, Dict[str, Any]]],
        raise_after: Optional[int] = None,
    ) -> None:
        self.events = events
        self.raise_after = raise_after

    async def stream_events(
        self,
        sandbox_url: str,
        opencode_session_id: str,
        our_session_id: str,
    ) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
        for i, evt in enumerate(self.events):
            if self.raise_after is not None and i >= self.raise_after:
                raise SandboxUnreachableError("upstream is gone")
            yield evt
        if self.raise_after is not None and self.raise_after >= len(self.events):
            raise SandboxUnreachableError("upstream is gone")


# ---------------------------------------------------------------------------
# Happy path — SSE format + headers
# ---------------------------------------------------------------------------


def test_stream_events_happy_path_format_and_headers() -> None:
    app = _build_app()
    client = TestClient(app)

    fake_adapter = _FakeAdapter(
        events=[
            ("connected", {"session_id": "ses_test"}),
            (
                "message.started",
                {"message_id": "msg_a", "role": "assistant"},
            ),
            (
                "message.completed",
                {
                    "message_id": "msg_a",
                    "content": "hi",
                    "completed_at": "2026-05-07T15:04:05.123Z",
                },
            ),
        ]
    )

    async def _fake_get_session(prisma_client, *, session_id, created_by):
        assert session_id == "ses_test"
        assert created_by == CALLER_USER_ID
        return _ready_session_row()

    with (
        patch(
            "litellm.proxy.managed_agents_endpoints.events.get_session",
            side_effect=_fake_get_session,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.events.get_adapter",
            return_value=fake_adapter,
        ),
        patch("litellm.proxy.proxy_server.prisma_client", object()),
    ):
        with client.stream("GET", "/v2/sessions/ses_test/events") as r:
            assert r.status_code == 200
            # Headers per contract §6.6.
            assert r.headers["content-type"].startswith("text/event-stream")
            assert r.headers["cache-control"] == "no-cache"
            assert r.headers["connection"] == "keep-alive"
            assert r.headers["x-accel-buffering"] == "no"
            body = b"".join(r.iter_bytes()).decode()

    # First event must be `connected` per contract.
    assert body.startswith('event: connected\ndata: {"session_id": "ses_test"}\n\n')
    # Each event in order, separated by blank line.
    assert (
        "event: message.started\n"
        'data: {"message_id": "msg_a", "role": "assistant"}\n\n'
    ) in body
    assert "event: message.completed\n" in body
    assert body.endswith("\n\n")


# ---------------------------------------------------------------------------
# 404 — session missing (status code returned BEFORE streaming starts)
# ---------------------------------------------------------------------------


def test_stream_events_404_when_session_missing() -> None:
    app = _build_app()
    client = TestClient(app)

    async def _fake_get_session(prisma_client, *, session_id, created_by):
        return None

    with (
        patch(
            "litellm.proxy.managed_agents_endpoints.events.get_session",
            side_effect=_fake_get_session,
        ),
        patch("litellm.proxy.proxy_server.prisma_client", object()),
    ):
        r = client.get("/v2/sessions/ses_missing/events")

    assert r.status_code == 404
    assert "ses_missing" in r.json()["detail"]
    assert r.headers["content-type"].startswith("application/json")


# ---------------------------------------------------------------------------
# 503 — session is still provisioning (status code BEFORE streaming starts)
# ---------------------------------------------------------------------------


def test_stream_events_503_when_session_provisioning() -> None:
    app = _build_app()
    client = TestClient(app)

    provisioning_row = _ready_session_row()
    provisioning_row["status"] = "provisioning"

    async def _fake_get_session(prisma_client, *, session_id, created_by):
        return provisioning_row

    with (
        patch(
            "litellm.proxy.managed_agents_endpoints.events.get_session",
            side_effect=_fake_get_session,
        ),
        patch("litellm.proxy.proxy_server.prisma_client", object()),
    ):
        r = client.get("/v2/sessions/ses_test/events")

    assert r.status_code == 503
    assert r.json()["detail"] == "Session not ready"
    # Retry-After hint per contract §7 failure modes table.
    assert r.headers.get("retry-after") == "5"


# ---------------------------------------------------------------------------
# 404 — terminated session is treated as not-found (per contract §7)
# ---------------------------------------------------------------------------


def test_stream_events_404_when_session_terminated() -> None:
    app = _build_app()
    client = TestClient(app)

    terminated_row = _ready_session_row()
    terminated_row["status"] = "terminated"

    async def _fake_get_session(prisma_client, *, session_id, created_by):
        return terminated_row

    with (
        patch(
            "litellm.proxy.managed_agents_endpoints.events.get_session",
            side_effect=_fake_get_session,
        ),
        patch("litellm.proxy.proxy_server.prisma_client", object()),
    ):
        r = client.get("/v2/sessions/ses_test/events")

    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Mid-stream SandboxUnreachableError → inline `error` event (status is 200)
# ---------------------------------------------------------------------------


def test_stream_events_unreachable_after_stream_started_emits_error_event() -> None:
    app = _build_app()
    client = TestClient(app)

    fake_adapter = _FakeAdapter(
        events=[
            ("connected", {"session_id": "ses_test"}),
            ("message.started", {"message_id": "msg_a", "role": "assistant"}),
        ],
        raise_after=2,  # raise immediately after both events stream.
    )

    async def _fake_get_session(prisma_client, *, session_id, created_by):
        return _ready_session_row()

    with (
        patch(
            "litellm.proxy.managed_agents_endpoints.events.get_session",
            side_effect=_fake_get_session,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.events.get_adapter",
            return_value=fake_adapter,
        ),
        patch("litellm.proxy.proxy_server.prisma_client", object()),
    ):
        with client.stream("GET", "/v2/sessions/ses_test/events") as r:
            # Status was already 200 by the time the adapter raised.
            assert r.status_code == 200
            body = b"".join(r.iter_bytes()).decode()

    assert "event: connected\n" in body
    assert "event: message.started\n" in body
    # Error becomes an inline SSE event with the canonical message.
    assert 'event: error\ndata: {"error": "Sandbox unreachable"}\n\n' in body


# ---------------------------------------------------------------------------
# Pre-stream SandboxUnreachableError → still 200 with error event
# (the adapter raises on the very first iteration, before any events stream).
# ---------------------------------------------------------------------------


def test_stream_events_unreachable_on_first_iteration_emits_error_only() -> None:
    app = _build_app()
    client = TestClient(app)

    # raise_after=0 means: raise before yielding anything.
    fake_adapter = _FakeAdapter(events=[], raise_after=0)

    async def _fake_get_session(prisma_client, *, session_id, created_by):
        return _ready_session_row()

    with (
        patch(
            "litellm.proxy.managed_agents_endpoints.events.get_session",
            side_effect=_fake_get_session,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.events.get_adapter",
            return_value=fake_adapter,
        ),
        patch("litellm.proxy.proxy_server.prisma_client", object()),
    ):
        with client.stream("GET", "/v2/sessions/ses_test/events") as r:
            # StreamingResponse always returns 200 once it's been constructed —
            # the streaming generator hasn't run yet at that point.
            assert r.status_code == 200
            body = b"".join(r.iter_bytes()).decode()

    assert body == 'event: error\ndata: {"error": "Sandbox unreachable"}\n\n'


# ---------------------------------------------------------------------------
# 500 — corrupt session row (missing sandbox state)
# ---------------------------------------------------------------------------


def test_stream_events_500_when_session_row_missing_sandbox_state() -> None:
    app = _build_app()
    client = TestClient(app)

    bad_row = _ready_session_row()
    bad_row["sandbox_url"] = None  # corrupt: ready session must have a url.

    async def _fake_get_session(prisma_client, *, session_id, created_by):
        return bad_row

    with (
        patch(
            "litellm.proxy.managed_agents_endpoints.events.get_session",
            side_effect=_fake_get_session,
        ),
        patch("litellm.proxy.proxy_server.prisma_client", object()),
    ):
        r = client.get("/v2/sessions/ses_test/events")

    assert r.status_code == 500


# ---------------------------------------------------------------------------
# JSON-string sandbox_metadata is parsed defensively
# ---------------------------------------------------------------------------


def test_stream_events_handles_json_string_sandbox_metadata() -> None:
    """Some Prisma clients return JSON columns as strings — handle that."""
    app = _build_app()
    client = TestClient(app)

    row = _ready_session_row()
    row["sandbox_metadata"] = '{"opencode_session_id": "oc_sid_abc"}'

    fake_adapter = _FakeAdapter(events=[("connected", {"session_id": "ses_test"})])

    async def _fake_get_session(prisma_client, *, session_id, created_by):
        return row

    with (
        patch(
            "litellm.proxy.managed_agents_endpoints.events.get_session",
            side_effect=_fake_get_session,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.events.get_adapter",
            return_value=fake_adapter,
        ),
        patch("litellm.proxy.proxy_server.prisma_client", object()),
    ):
        with client.stream("GET", "/v2/sessions/ses_test/events") as r:
            assert r.status_code == 200
            body = b"".join(r.iter_bytes()).decode()

    assert "event: connected\n" in body


# ---------------------------------------------------------------------------
# 500 — prisma client not initialized
# ---------------------------------------------------------------------------


def test_stream_events_500_when_prisma_client_not_initialized() -> None:
    app = _build_app()
    client = TestClient(app)

    with patch("litellm.proxy.proxy_server.prisma_client", None):
        r = client.get("/v2/sessions/ses_test/events")

    assert r.status_code == 500


# ---------------------------------------------------------------------------
# Caller scoping is enforced — _load_ready_session passes user_id through.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_events_scopes_by_caller_user_id() -> None:
    """The handler must pass `user_api_key_dict.user_id` as `created_by`.

    Without scoping, callers could read other users' sessions.
    """
    app = _build_app()
    client = TestClient(app)

    captured: Dict[str, Any] = {}

    async def _fake_get_session(prisma_client, *, session_id, created_by):
        captured["session_id"] = session_id
        captured["created_by"] = created_by
        return None  # 404 — but we only care that scoping was used.

    with (
        patch(
            "litellm.proxy.managed_agents_endpoints.events.get_session",
            side_effect=_fake_get_session,
        ),
        patch("litellm.proxy.proxy_server.prisma_client", object()),
    ):
        r = client.get("/v2/sessions/ses_test/events")

    assert r.status_code == 404
    assert captured["session_id"] == "ses_test"
    assert captured["created_by"] == CALLER_USER_ID
