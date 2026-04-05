"""
Verify that timeout is forwarded through make_call() (async) and
make_sync_call() (sync) to client.post() for Vertex AI Gemini streaming requests.

Regression test for https://github.com/BerriAI/litellm/issues/23375
"""

import asyncio
import os
import sys
from functools import partial
from unittest.mock import AsyncMock, MagicMock

import httpx

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)


def _run_vertex_make_call(**extra_kwargs):
    """Helper to call vertex make_call with mocked dependencies."""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        make_call,
    )

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = MagicMock(return_value=AsyncMock())

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_logging = MagicMock()

    asyncio.run(
        make_call(
            client=mock_client,
            gemini_client=None,
            api_base="https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/google/models/gemini:streamGenerateContent",
            headers={"Authorization": "Bearer token"},
            data='{"contents": []}',
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "test"}],
            logging_obj=mock_logging,
            **extra_kwargs,
        )
    )
    return mock_client


def test_vertex_make_call_forwards_timeout_to_client_post():
    mock_client = _run_vertex_make_call(timeout=0.1)
    mock_client.post.assert_called_once()
    assert mock_client.post.call_args.kwargs.get("timeout") == 0.1


def test_vertex_make_call_timeout_defaults_to_none():
    mock_client = _run_vertex_make_call()
    assert mock_client.post.call_args.kwargs.get("timeout") is None


def test_vertex_make_call_forwards_httpx_timeout_object():
    timeout_obj = httpx.Timeout(5.0, connect=2.0)
    mock_client = _run_vertex_make_call(timeout=timeout_obj)
    assert mock_client.post.call_args.kwargs.get("timeout") is timeout_obj


def test_vertex_make_call_partial_includes_timeout():
    """Verify that partial(make_call, ..., timeout=X) binds the timeout arg."""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        make_call,
    )

    bound = partial(
        make_call,
        gemini_client=None,
        api_base="https://example.com",
        headers={},
        data="{}",
        model="test",
        messages=[],
        logging_obj=MagicMock(),
        timeout=0.5,
    )
    assert bound.keywords["timeout"] == 0.5


# --- Sync path: make_sync_call ---


def _run_vertex_make_sync_call(**extra_kwargs):
    """Helper to call vertex make_sync_call with mocked dependencies."""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        make_sync_call,
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines = MagicMock(return_value=iter([]))

    mock_client = MagicMock()
    mock_client.post = MagicMock(return_value=mock_response)

    mock_logging = MagicMock()

    make_sync_call(
        client=None,
        gemini_client=mock_client,
        api_base="https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/google/models/gemini:streamGenerateContent",
        headers={"Authorization": "Bearer token"},
        data='{"contents": []}',
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "test"}],
        logging_obj=mock_logging,
        **extra_kwargs,
    )
    return mock_client


def test_vertex_make_sync_call_forwards_timeout_to_client_post():
    mock_client = _run_vertex_make_sync_call(timeout=0.1)
    mock_client.post.assert_called_once()
    assert mock_client.post.call_args.kwargs.get("timeout") == 0.1


def test_vertex_make_sync_call_timeout_defaults_to_none():
    mock_client = _run_vertex_make_sync_call()
    assert mock_client.post.call_args.kwargs.get("timeout") is None


def test_vertex_make_sync_call_forwards_httpx_timeout_object():
    timeout_obj = httpx.Timeout(5.0, connect=2.0)
    mock_client = _run_vertex_make_sync_call(timeout=timeout_obj)
    assert mock_client.post.call_args.kwargs.get("timeout") is timeout_obj


def test_vertex_make_sync_call_partial_includes_timeout():
    """Verify that partial(make_sync_call, ..., timeout=X) binds the timeout arg."""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        make_sync_call,
    )

    bound = partial(
        make_sync_call,
        gemini_client=None,
        api_base="https://example.com",
        headers={},
        data="{}",
        model="test",
        messages=[],
        logging_obj=MagicMock(),
        timeout=0.5,
    )
    assert bound.keywords["timeout"] == 0.5
