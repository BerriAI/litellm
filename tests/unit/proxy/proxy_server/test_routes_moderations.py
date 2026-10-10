"""Behavior pins for ``proxy_server.py`` moderations routes.

Pins (PR2):
    - POST /v1/moderations
    - POST /moderations
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy import proxy_server

from .conftest import normalize  # type: ignore[import-not-found]

HAPPY_RESPONSE = {
    "id": "modr-test",
    "model": "text-moderation-stable",
    "results": [
        {
            "flagged": False,
            "categories": {"violence": False},
            "category_scores": {"violence": 0.01},
        }
    ],
}


@pytest.fixture
def patched_moderation(monkeypatch):
    monkeypatch.setattr(proxy_server, "llm_router", MagicMock())
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        MagicMock(
            pre_call_hook=AsyncMock(side_effect=lambda **kw: kw["data"]),
            post_call_failure_hook=AsyncMock(),
            update_request_status=AsyncMock(),
        ),
    )

    async def _add_data(data, **kwargs):
        return data

    monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", _add_data)

    async def _fake_llm_call():
        return dict(HAPPY_RESPONSE)

    async def _fake_route_request(*args, **kwargs):
        return _fake_llm_call()

    monkeypatch.setattr(proxy_server, "route_request", _fake_route_request)
    yield


@pytest.fixture
def moderation_pipeline_raises(monkeypatch):
    monkeypatch.setattr(proxy_server, "llm_router", MagicMock())
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        MagicMock(
            pre_call_hook=AsyncMock(side_effect=lambda **kw: kw["data"]),
            post_call_failure_hook=AsyncMock(),
            update_request_status=AsyncMock(),
        ),
    )

    async def _add_data(data, **kwargs):
        return data

    monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", _add_data)

    async def _raise(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(proxy_server, "route_request", _raise)
    yield


@pytest.mark.parametrize("path", ["/v1/moderations", "/moderations"])
def test_moderation_happy_path(client, auth_as, patched_moderation, path):
    """Pins ``POST /v1/moderations`` and ``POST /moderations`` (happy)."""
    payload = {"model": "text-moderation-stable", "input": "Sample text"}
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "id": "<VOLATILE>",
        "model": "text-moderation-stable",
        "results": [
            {
                "flagged": False,
                "categories": {"violence": False},
                "category_scores": {"violence": 0.01},
            }
        ],
    }


@pytest.mark.parametrize("path", ["/v1/moderations", "/moderations"])
def test_moderation_error(client, auth_as, moderation_pipeline_raises, path):
    """Pins ``POST /v1/moderations`` and ``POST /moderations`` (error)."""
    payload = {"model": "text-moderation-stable", "input": "Sample text"}
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 500
    assert len(response.content) > 0
