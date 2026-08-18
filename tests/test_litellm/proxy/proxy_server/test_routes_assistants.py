"""Behavior pins for ``proxy_server.py`` assistants routes.

Pins (PR2):
    - GET /v1/assistants
    - GET /assistants
    - POST /v1/assistants
    - POST /assistants
    - DELETE /v1/assistants/{assistant_id:path}
    - DELETE /assistants/{assistant_id:path}
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy import proxy_server

from .conftest import normalize  # type: ignore[import-not-found]

GET_RESPONSE = {
    "object": "list",
    "data": [
        {
            "id": "asst_1",
            "object": "assistant",
            "name": "Test Assistant",
            "model": "gpt-4",
        }
    ],
    "first_id": "asst_1",
    "last_id": "asst_1",
    "has_more": False,
}


CREATE_RESPONSE = {
    "id": "asst_new",
    "object": "assistant",
    "name": "New",
    "model": "gpt-4",
    "created_at": 0,
}


DELETE_RESPONSE = {"id": "asst_1", "object": "assistant.deleted", "deleted": True}


@pytest.fixture
def patched_assistants(monkeypatch):
    router = MagicMock()
    router.aget_assistants = AsyncMock(return_value=dict(GET_RESPONSE))
    router.acreate_assistants = AsyncMock(return_value=dict(CREATE_RESPONSE))
    router.adelete_assistant = AsyncMock(return_value=dict(DELETE_RESPONSE))
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
# GET /v1/assistants, GET /assistants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/v1/assistants", "/assistants"])
def test_get_assistants_happy_path(client, auth_as, patched_assistants, path):
    """Pins ``GET /v1/assistants`` and ``GET /assistants``."""
    with auth_as():
        response = client.get(path)
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "object": "list",
        "data": [
            {
                "id": "<VOLATILE>",
                "object": "assistant",
                "name": "Test Assistant",
                "model": "gpt-4",
            }
        ],
        "first_id": "asst_1",
        "last_id": "asst_1",
        "has_more": False,
    }


@pytest.mark.parametrize("path", ["/v1/assistants", "/assistants"])
def test_get_assistants_no_router_error(client, auth_as, no_router, path):
    """Pins ``GET /v1/assistants`` and ``GET /assistants`` (error: no llm_router)."""
    with auth_as():
        response = client.get(path)
    assert response.status_code == 500
    assert len(response.content) > 0


# ---------------------------------------------------------------------------
# POST /v1/assistants, POST /assistants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/v1/assistants", "/assistants"])
def test_create_assistant_happy_path(client, auth_as, patched_assistants, path):
    """Pins ``POST /v1/assistants`` and ``POST /assistants``."""
    payload = {"model": "gpt-4", "name": "New"}
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "id": "<VOLATILE>",
        "object": "assistant",
        "name": "New",
        "model": "gpt-4",
        "created_at": "<VOLATILE>",
    }


@pytest.mark.parametrize("path", ["/v1/assistants", "/assistants"])
def test_create_assistant_no_router_error(client, auth_as, no_router, path):
    """Pins ``POST /v1/assistants`` and ``POST /assistants`` (error: no llm_router)."""
    with auth_as():
        response = client.post(path, json={"model": "gpt-4"})
    assert response.status_code == 500
    assert len(response.content) > 0


# ---------------------------------------------------------------------------
# DELETE /v1/assistants/{assistant_id:path}, DELETE /assistants/{assistant_id:path}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/v1/assistants/asst_1", "/assistants/asst_1"])
def test_delete_assistant_happy_path(client, auth_as, patched_assistants, path):
    """Pins ``DELETE /v1/assistants/{assistant_id:path}`` and ``DELETE /assistants/{assistant_id:path}``."""
    with auth_as():
        response = client.delete(path)
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "id": "<VOLATILE>",
        "object": "assistant.deleted",
        "deleted": True,
    }


@pytest.mark.parametrize("path", ["/v1/assistants/asst_1", "/assistants/asst_1"])
def test_delete_assistant_no_router_error(client, auth_as, no_router, path):
    """Pins ``DELETE /v1/assistants/{assistant_id:path}`` / ``DELETE /assistants/{assistant_id:path}`` (error)."""
    with auth_as():
        response = client.delete(path)
    assert response.status_code == 500
    assert len(response.content) > 0
