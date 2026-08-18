"""Behavior pins for ``proxy_server.py`` threads routes.

Pins (PR2):
    - POST /v1/threads
    - POST /threads
    - GET /v1/threads/{thread_id}
    - GET /threads/{thread_id}
    - POST /v1/threads/{thread_id}/messages
    - POST /threads/{thread_id}/messages
    - GET /v1/threads/{thread_id}/messages
    - GET /threads/{thread_id}/messages
    - POST /v1/threads/{thread_id}/runs
    - POST /threads/{thread_id}/runs
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy import proxy_server

from .conftest import normalize  # type: ignore[import-not-found]

CREATE_THREAD = {"id": "thr_1", "object": "thread", "created_at": 0, "metadata": {}}
GET_THREAD = {
    "id": "thr_1",
    "object": "thread",
    "created_at": 0,
    "tool_resources": {},
}
ADD_MESSAGE = {
    "id": "msg_1",
    "object": "thread.message",
    "thread_id": "thr_1",
    "role": "user",
    "content": [],
}
GET_MESSAGES = {
    "object": "list",
    "data": [
        {
            "id": "msg_1",
            "object": "thread.message",
            "thread_id": "thr_1",
            "role": "user",
            "content": [],
        }
    ],
    "first_id": "msg_1",
    "last_id": "msg_1",
    "has_more": False,
}
RUN_THREAD = {
    "id": "run_1",
    "object": "thread.run",
    "thread_id": "thr_1",
    "assistant_id": "asst_1",
    "status": "queued",
}


@pytest.fixture
def patched_threads(monkeypatch):
    router = MagicMock()
    router.acreate_thread = AsyncMock(return_value=dict(CREATE_THREAD))
    router.aget_thread = AsyncMock(return_value=dict(GET_THREAD))
    router.a_add_message = AsyncMock(return_value=dict(ADD_MESSAGE))
    router.aget_messages = AsyncMock(return_value=dict(GET_MESSAGES))
    router.arun_thread = AsyncMock(return_value=dict(RUN_THREAD))
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        MagicMock(
            post_call_failure_hook=AsyncMock(), update_request_status=AsyncMock()
        ),
    )

    async def _add_data(data, **kwargs):
        return data

    monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", _add_data)
    return router


@pytest.fixture
def no_router(monkeypatch):
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        MagicMock(
            post_call_failure_hook=AsyncMock(), update_request_status=AsyncMock()
        ),
    )

    async def _add_data(data, **kwargs):
        return data

    monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", _add_data)
    yield


# ---------------------------------------------------------------------------
# POST /v1/threads, POST /threads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/v1/threads", "/threads"])
def test_create_thread_happy(client, auth_as, patched_threads, path):
    """Pins ``POST /v1/threads`` and ``POST /threads``."""
    with auth_as():
        response = client.post(path, json={})
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "id": "<VOLATILE>",
        "object": "thread",
        "created_at": "<VOLATILE>",
        "metadata": {},
    }


@pytest.mark.parametrize("path", ["/v1/threads", "/threads"])
def test_create_thread_error(client, auth_as, no_router, path):
    """Pins ``POST /v1/threads`` / ``POST /threads`` (error: no llm_router)."""
    with auth_as():
        response = client.post(path, json={})
    assert response.status_code == 500
    assert len(response.content) > 0


# ---------------------------------------------------------------------------
# GET /v1/threads/{thread_id}, GET /threads/{thread_id}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/v1/threads/thr_1", "/threads/thr_1"])
def test_get_thread_happy(client, auth_as, patched_threads, path):
    """Pins ``GET /v1/threads/{thread_id}`` and ``GET /threads/{thread_id}``."""
    with auth_as():
        response = client.get(path)
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "id": "<VOLATILE>",
        "object": "thread",
        "created_at": "<VOLATILE>",
        "tool_resources": {},
    }


@pytest.mark.parametrize("path", ["/v1/threads/thr_1", "/threads/thr_1"])
def test_get_thread_error(client, auth_as, no_router, path):
    """Pins ``GET /v1/threads/{thread_id}`` / ``GET /threads/{thread_id}`` (error)."""
    with auth_as():
        response = client.get(path)
    assert response.status_code == 500
    assert len(response.content) > 0


# ---------------------------------------------------------------------------
# POST /v1/threads/{thread_id}/messages, POST /threads/{thread_id}/messages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/v1/threads/thr_1/messages", "/threads/thr_1/messages"],
)
def test_add_message_happy(client, auth_as, patched_threads, path):
    """Pins ``POST /v1/threads/{thread_id}/messages`` and ``POST /threads/{thread_id}/messages``."""
    payload = {"role": "user", "content": "hi"}
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "id": "<VOLATILE>",
        "object": "thread.message",
        "thread_id": "thr_1",
        "role": "user",
        "content": [],
    }


@pytest.mark.parametrize(
    "path",
    ["/v1/threads/thr_1/messages", "/threads/thr_1/messages"],
)
def test_add_message_error(client, auth_as, no_router, path):
    """Pins ``POST /v1/threads/{thread_id}/messages`` / ``POST /threads/{thread_id}/messages`` (error)."""
    with auth_as():
        response = client.post(path, json={"role": "user", "content": "hi"})
    assert response.status_code == 500
    assert len(response.content) > 0


# ---------------------------------------------------------------------------
# GET /v1/threads/{thread_id}/messages, GET /threads/{thread_id}/messages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/v1/threads/thr_1/messages", "/threads/thr_1/messages"],
)
def test_get_messages_happy(client, auth_as, patched_threads, path):
    """Pins ``GET /v1/threads/{thread_id}/messages`` and ``GET /threads/{thread_id}/messages``."""
    with auth_as():
        response = client.get(path)
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "object": "list",
        "data": [
            {
                "id": "<VOLATILE>",
                "object": "thread.message",
                "thread_id": "thr_1",
                "role": "user",
                "content": [],
            }
        ],
        "first_id": "msg_1",
        "last_id": "msg_1",
        "has_more": False,
    }


@pytest.mark.parametrize(
    "path",
    ["/v1/threads/thr_1/messages", "/threads/thr_1/messages"],
)
def test_get_messages_error(client, auth_as, no_router, path):
    """Pins ``GET /v1/threads/{thread_id}/messages`` / ``GET /threads/{thread_id}/messages`` (error)."""
    with auth_as():
        response = client.get(path)
    assert response.status_code == 500
    assert len(response.content) > 0


# ---------------------------------------------------------------------------
# POST /v1/threads/{thread_id}/runs, POST /threads/{thread_id}/runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/v1/threads/thr_1/runs", "/threads/thr_1/runs"],
)
def test_run_thread_happy(client, auth_as, patched_threads, path):
    """Pins ``POST /v1/threads/{thread_id}/runs`` and ``POST /threads/{thread_id}/runs``."""
    payload = {"assistant_id": "asst_1"}
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "id": "<VOLATILE>",
        "object": "thread.run",
        "thread_id": "thr_1",
        "assistant_id": "asst_1",
        "status": "queued",
    }


@pytest.mark.parametrize(
    "path",
    ["/v1/threads/thr_1/runs", "/threads/thr_1/runs"],
)
def test_run_thread_error(client, auth_as, no_router, path):
    """Pins ``POST /v1/threads/{thread_id}/runs`` / ``POST /threads/{thread_id}/runs`` (error)."""
    with auth_as():
        response = client.post(path, json={"assistant_id": "asst_1"})
    assert response.status_code == 500
    assert len(response.content) > 0
