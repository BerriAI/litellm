"""
End-to-end tests for Vertex AI rerank `userLabels` propagation.

These tests go through the full `litellm.rerank()` call path with the HTTP
layer mocked, so they catch plumbing bugs (e.g. `litellm_params` losing
`metadata` between the rerank entrypoint and the Vertex transform) that
unit tests on `VertexAIRerankConfig.transform_rerank_request` miss.
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
import litellm.llms.vertex_ai.rerank.transformation


def _extract_body(call_kwargs):
    """The rerank handler sends `data=json.dumps(...)`, not `json=...`."""
    if "json" in call_kwargs and call_kwargs["json"] is not None:
        return call_kwargs["json"]
    raw = call_kwargs.get("data")
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw) if raw else None


def _make_mock_rank_response():
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "records": [
            {"id": "0", "score": 0.9, "title": "doc 0", "content": "hello"},
            {"id": "1", "score": 0.1, "title": "doc 1", "content": "world"},
        ]
    }
    mock_response.text = '{"records": []}'
    return mock_response


def _make_async_mock_rank_response():
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json = MagicMock(
        return_value={
            "records": [
                {"id": "0", "score": 0.9, "title": "doc 0", "content": "hello"},
            ]
        }
    )
    mock_response.text = '{"records": []}'
    return mock_response


@pytest.fixture
def clean_vertex_env():
    saved = {}
    for var in (
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "VERTEXAI_PROJECT",
        "VERTEXAI_CREDENTIALS",
        "VERTEX_AI_CREDENTIALS",
        "VERTEX_PROJECT",
        "VERTEX_LOCATION",
        "VERTEX_AI_PROJECT",
    ):
        if var in os.environ:
            saved[var] = os.environ.pop(var)
    yield
    for var, value in saved.items():
        os.environ[var] = value


def _patch_vertex_auth():
    return patch.object(
        litellm.llms.vertex_ai.rerank.transformation.VertexAIRerankConfig,
        "_ensure_access_token",
        return_value=("test-access-token", "test-project-2049"),
    )


def test_rerank_userlabels_propagates_from_metadata_sync(clean_vertex_env):
    """
    `litellm.rerank(metadata={"requester_metadata": {...}})` must end up as
    `userLabels` on the Discovery Engine `:rank` request body.
    """
    captured = {}

    def fake_post(*args, **kwargs):
        captured["body"] = _extract_body(kwargs)
        return _make_mock_rank_response()

    with (
        _patch_vertex_auth(),
        patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            side_effect=fake_post,
        ),
    ):
        litellm.rerank(
            model="vertex_ai/semantic-ranker-default@latest",
            query="what is gemini?",
            documents=["hello", "world"],
            vertex_project="test-project-2049",
            vertex_credentials='{"type": "service_account"}',
            metadata={"requester_metadata": {"team": "platform", "env": "prod"}},
        )

    body = captured["body"]
    assert body is not None, "expected POST body to be captured"
    assert "userLabels" in body, (
        "Vertex rerank request body is missing `userLabels` — metadata was "
        "lost between litellm.rerank() and transform_rerank_request. "
        f"body keys: {sorted(body.keys())}"
    )
    assert body["userLabels"] == {"team": "platform", "env": "prod"}


def test_rerank_userlabels_propagates_from_metadata_async(clean_vertex_env):
    """Same as the sync test, but through `litellm.arerank`."""
    captured = {}

    async def fake_post(*args, **kwargs):
        captured["body"] = _extract_body(kwargs)
        return _make_async_mock_rank_response()

    with (
        _patch_vertex_auth(),
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            side_effect=fake_post,
        ),
    ):
        asyncio.run(
            litellm.arerank(
                model="vertex_ai/semantic-ranker-default@latest",
                query="what is gemini?",
                documents=["hello", "world"],
                vertex_project="test-project-2049",
                vertex_credentials='{"type": "service_account"}',
                metadata={"requester_metadata": {"team": "platform"}},
            )
        )

    body = captured["body"]
    assert body is not None
    assert body.get("userLabels") == {"team": "platform"}


def test_rerank_userlabels_absent_when_no_metadata(clean_vertex_env):
    """No metadata → no `userLabels` key (don't send empty maps)."""
    captured = {}

    def fake_post(*args, **kwargs):
        captured["body"] = _extract_body(kwargs)
        return _make_mock_rank_response()

    with (
        _patch_vertex_auth(),
        patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            side_effect=fake_post,
        ),
    ):
        litellm.rerank(
            model="vertex_ai/semantic-ranker-default@latest",
            query="what is gemini?",
            documents=["hello", "world"],
            vertex_project="test-project-2049",
            vertex_credentials='{"type": "service_account"}',
        )

    body = captured["body"]
    assert body is not None
    assert "userLabels" not in body
