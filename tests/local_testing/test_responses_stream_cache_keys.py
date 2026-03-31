from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm import aresponses
from litellm._uuid import uuid
from litellm.caching.caching_handler import LLMCachingHandler
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.types.utils import CallTypes


@pytest.mark.asyncio
async def test_async_get_cache_reuses_preset_cache_key_for_responses():
    caching_handler = LLMCachingHandler(
        original_function=aresponses,
        request_kwargs={},
        start_time=datetime.now(),
    )
    logging_obj = LiteLLMLogging(
        litellm_call_id=str(datetime.now()),
        call_type=CallTypes.aresponses.value,
        model="gpt-4.1-mini",
        messages=[],
        function_id=str(uuid.uuid4()),
        stream=True,
        start_time=datetime.now(),
    )

    original_cache = litellm.cache
    mock_cache = MagicMock()
    mock_cache.supported_call_types = [CallTypes.aresponses.value]
    mock_cache._supports_async.return_value = True
    mock_cache.get_cache_key.return_value = "responses-stream-cache-key"
    mock_cache.async_get_cache = AsyncMock(return_value=None)
    litellm.cache = mock_cache

    kwargs = {
        "model": "gpt-4.1-mini",
        "input": "hello",
        "stream": True,
        "litellm_params": {},
    }
    await caching_handler._async_get_cache(
        model="gpt-4.1-mini",
        original_function=aresponses,
        logging_obj=logging_obj,
        start_time=datetime.now(),
        call_type=CallTypes.aresponses.value,
        kwargs=kwargs,
    )

    assert caching_handler.preset_cache_key == "responses-stream-cache-key"
    mock_cache.async_get_cache.assert_awaited_once()
    assert (
        mock_cache.async_get_cache.call_args.kwargs["cache_key"]
        == "responses-stream-cache-key"
    )

    litellm.cache = original_cache
