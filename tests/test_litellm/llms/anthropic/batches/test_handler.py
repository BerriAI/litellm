"""
Tests for AnthropicBatchesHandler list_batches and cancel_batch

Same seam discipline as tests/unit/llms/anthropic/batches/test_handler.py:
only the async httpx client and credential resolvers are mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.llms.anthropic.batches.handler import AnthropicBatchesHandler
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.types.utils import LiteLLMBatch


@pytest.fixture
def handler():
    return AnthropicBatchesHandler()


def _list_response():
    return httpx.Response(
        status_code=200,
        json={
            "data": [{"id": "msgbatch_1", "processing_status": "ended", "request_counts": {"succeeded": 1}}],
            "has_more": False,
            "first_id": "msgbatch_1",
            "last_id": "msgbatch_1",
        },
        request=httpx.Request("GET", "https://api.anthropic.com/v1/messages/batches"),
    )


def _cancel_response():
    return httpx.Response(
        status_code=200,
        json={
            "id": "msgbatch_abc",
            "processing_status": "canceling",
            "created_at": "2024-09-24T10:00:00Z",
            "cancel_initiated_at": "2024-09-24T10:30:00Z",
            "request_counts": {},
        },
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages/batches/msgbatch_abc/cancel"),
    )


@pytest.fixture
def patched_client():
    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=_list_response())
    fake_client.post = AsyncMock(return_value=_cancel_response())
    with patch(
        "litellm.llms.anthropic.batches.handler.get_async_httpx_client",
        return_value=fake_client,
    ):
        yield fake_client


@pytest.mark.asyncio
async def test_alist_batches_gets_list_url_with_pagination_and_headers(handler, patched_client):
    result = await handler.alist_batches(
        after="msgbatch_cursor",
        limit=10,
        api_base="https://api.anthropic.com",
        api_key="sk-ant-test",
        timeout=60.0,
        max_retries=0,
    )

    fake_client = patched_client
    fake_client.get.assert_awaited_once()
    _, call_kwargs = fake_client.get.call_args
    url = httpx.URL(call_kwargs["url"])
    assert url.path == "/v1/messages/batches"
    assert url.params["limit"] == "10"
    assert url.params["after_id"] == "msgbatch_cursor"
    headers = call_kwargs["headers"]
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["anthropic-beta"] == "message-batches-2024-09-24"
    assert call_kwargs["timeout"] == 60.0

    assert result["object"] == "list"
    assert result["has_more"] is False
    assert [b.id for b in result["data"]] == ["msgbatch_1"]


@pytest.mark.asyncio
async def test_acancel_batch_posts_to_cancel_url_and_maps_response(handler, patched_client):
    batch = await handler.acancel_batch(
        batch_id="msgbatch_abc",
        api_base="https://api.anthropic.com",
        api_key="sk-ant-test",
        timeout=60.0,
        max_retries=0,
    )

    fake_client = patched_client
    fake_client.post.assert_awaited_once()
    _, call_kwargs = fake_client.post.call_args
    assert call_kwargs["url"] == "https://api.anthropic.com/v1/messages/batches/msgbatch_abc/cancel"
    assert call_kwargs["headers"]["x-api-key"] == "sk-ant-test"
    assert call_kwargs["timeout"] == 60.0

    assert isinstance(batch, LiteLLMBatch)
    assert batch.id == "msgbatch_abc"
    assert batch.status == "cancelling"


@pytest.mark.asyncio
async def test_alist_batches_4xx_raises_anthropic_error(handler):
    error_response = httpx.Response(
        status_code=401,
        json={"error": {"type": "authentication_error", "message": "bad key"}},
        request=httpx.Request("GET", "https://api.anthropic.com/v1/messages/batches"),
    )
    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=error_response)
    with patch(
        "litellm.llms.anthropic.batches.handler.get_async_httpx_client",
        return_value=fake_client,
    ):
        with pytest.raises(AnthropicError) as exc_info:
            await handler.alist_batches(
                after=None,
                limit=None,
                api_base="https://api.anthropic.com",
                api_key="sk-ant-test",
                timeout=60.0,
                max_retries=0,
            )
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_acancel_batch_4xx_raises_anthropic_error(handler):
    error_response = httpx.Response(
        status_code=404,
        json={"error": {"type": "not_found_error", "message": "no batch"}},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages/batches/missing/cancel"),
    )
    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=error_response)
    with patch(
        "litellm.llms.anthropic.batches.handler.get_async_httpx_client",
        return_value=fake_client,
    ):
        with pytest.raises(AnthropicError) as exc_info:
            await handler.acancel_batch(
                batch_id="missing",
                api_base="https://api.anthropic.com",
                api_key="sk-ant-test",
                timeout=60.0,
                max_retries=0,
            )
    assert exc_info.value.status_code == 404


def test_list_batches_sync_dispatch(handler, patched_client):
    result = handler.list_batches(
        _is_async=False,
        after=None,
        limit=5,
        api_base="https://api.anthropic.com",
        api_key="sk-ant-test",
        timeout=60.0,
        max_retries=0,
    )
    assert result["object"] == "list"


def test_cancel_batch_sync_dispatch(handler, patched_client):
    batch = handler.cancel_batch(
        _is_async=False,
        batch_id="msgbatch_abc",
        api_base="https://api.anthropic.com",
        api_key="sk-ant-test",
        timeout=60.0,
        max_retries=0,
    )
    assert batch.id == "msgbatch_abc"
