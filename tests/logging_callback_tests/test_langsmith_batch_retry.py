from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.integrations.langsmith import LangsmithLogger, LangsmithQueueObject


@pytest.mark.asyncio
async def test_flush_queue_preserves_only_failed_langsmith_credential_group():
    logger = LangsmithLogger(langsmith_api_key="test-key")
    successful_queue_obj = LangsmithQueueObject(
        data={"test": "data1"},
        credentials={
            "LANGSMITH_API_KEY": "key1",
            "LANGSMITH_PROJECT": "proj1",
            "LANGSMITH_BASE_URL": "url1",
            "LANGSMITH_TENANT_ID": None,
        },
    )
    failed_queue_obj = LangsmithQueueObject(
        data={"test": "data2"},
        credentials={
            "LANGSMITH_API_KEY": "key2",
            "LANGSMITH_PROJECT": "proj2",
            "LANGSMITH_BASE_URL": "url2",
            "LANGSMITH_TENANT_ID": None,
        },
    )
    logger.log_queue = [successful_queue_obj, failed_queue_obj]

    mock_success_response = MagicMock()
    mock_success_response.status_code = 200
    mock_success_response.raise_for_status = MagicMock()
    logger.async_httpx_client = MagicMock()
    logger.async_httpx_client.post = AsyncMock(
        side_effect=[mock_success_response, RuntimeError("send failed")]
    )

    await logger.flush_queue()

    assert logger.log_queue == [failed_queue_obj]
    assert logger.async_httpx_client.post.await_count == 2
