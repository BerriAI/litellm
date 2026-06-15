import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.openai.common_utils import BaseOpenAILLM

# Test parameters for different API functions
API_FUNCTION_PARAMS = [
    # (function_name, is_async, args)
    (
        "completion",
        False,
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        },
    ),
    (
        "completion",
        True,
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        },
    ),
    (
        "completion",
        True,
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": True,
        },
    ),
    ("embedding", False, {"model": "text-embedding-ada-002", "input": "Hello world"}),
    ("embedding", True, {"model": "text-embedding-ada-002", "input": "Hello world"}),
    (
        "image_generation",
        False,
        {"model": "dall-e-3", "prompt": "A beautiful sunset over mountains"},
    ),
    (
        "image_generation",
        True,
        {"model": "dall-e-3", "prompt": "A beautiful sunset over mountains"},
    ),
    (
        "speech",
        False,
        {
            "model": "tts-1",
            "input": "Hello, this is a test of text to speech",
            "voice": "alloy",
        },
    ),
    (
        "speech",
        True,
        {
            "model": "tts-1",
            "input": "Hello, this is a test of text to speech",
            "voice": "alloy",
        },
    ),
    ("transcription", False, {"model": "whisper-1", "file": MagicMock()}),
    ("transcription", True, {"model": "whisper-1", "file": MagicMock()}),
]


@pytest.mark.parametrize("function_name,is_async,args", API_FUNCTION_PARAMS)
@pytest.mark.asyncio
async def test_openai_client_reuse(function_name, is_async, args):
    """
    Test that multiple API calls reuse the same OpenAI client
    """
    litellm.set_verbose = True

    # Determine which client class to mock based on whether the test is async
    client_path = (
        "litellm.llms.openai.openai.AsyncOpenAI"
        if is_async
        else "litellm.llms.openai.openai.OpenAI"
    )

    # Create the appropriate patches
    with (
        patch(client_path) as mock_client_class,
        patch.object(BaseOpenAILLM, "set_cached_openai_client") as mock_set_cache,
        patch.object(BaseOpenAILLM, "get_cached_openai_client") as mock_get_cache,
    ):
        # Setup the mock to return None first time (cache miss) then a client for subsequent calls
        mock_client = MagicMock()
        mock_get_cache.side_effect = [None] + [
            mock_client
        ] * 9  # First call returns None, rest return the mock client

        # Make 10 API calls
        for _ in range(10):
            try:
                # Call the appropriate function based on parameters
                if is_async:
                    # Add 'a' prefix for async functions
                    func = getattr(litellm, f"a{function_name}")
                    await func(**args)
                else:
                    func = getattr(litellm, function_name)
                    func(**args)
            except Exception:
                # We expect exceptions since we're mocking the client
                pass

        # Verify client was created only once
        assert (
            mock_client_class.call_count == 1
        ), f"{'Async' if is_async else ''}OpenAI client should be created only once"

        # Verify the client was cached
        assert mock_set_cache.call_count == 1, "Client should be cached once"

        # Verify we tried to get from cache 10 times (once per request)
        assert mock_get_cache.call_count == 10, "Should check cache for each request"


def test_precomputed_init_params_match_inspect_signature():
    """
    Verify that the pre-computed _OPENAI_INIT_PARAMS and _AZURE_OPENAI_INIT_PARAMS
    match what inspect.signature() returns. If the OpenAI SDK changes its __init__
    params, this test will fail — signaling the constants need updating.
    """
    import inspect

    from openai import AzureOpenAI, OpenAI

    from litellm.llms.openai.common_utils import (
        _AZURE_OPENAI_INIT_PARAMS,
        _OPENAI_INIT_PARAMS,
    )

    expected_openai = tuple(
        p for p in inspect.signature(OpenAI.__init__).parameters if p != "self"
    )
    expected_azure = tuple(
        p for p in inspect.signature(AzureOpenAI.__init__).parameters if p != "self"
    )

    assert _OPENAI_INIT_PARAMS == expected_openai
    assert _AZURE_OPENAI_INIT_PARAMS == expected_azure


@pytest.mark.parametrize("client_type", ["openai", "azure"])
def test_get_openai_client_initialization_param_fields(client_type):
    """Verify the method returns the correct pre-computed params for each client type."""
    result = BaseOpenAILLM.get_openai_client_initialization_param_fields(client_type)
    assert isinstance(result, tuple)
    assert len(result) > 0
    assert "self" not in result


@pytest.mark.parametrize("client_type", ["openai", "azure"])
def test_get_openai_client_cache_key(client_type):
    """Verify get_openai_client_cache_key doesn't raise on tuple + tuple concatenation."""
    key = BaseOpenAILLM.get_openai_client_cache_key(
        client_initialization_params={"api_key": "sk-test"},
        client_type=client_type,
    )
    assert isinstance(key, str)
    assert "api_key=sk-test" in key


# ---------------------------------------------------------------------------
# Outbound HTTP/2 on the OpenAI/Azure client builders
# ---------------------------------------------------------------------------
class TestOpenAIClientHttp2:
    @pytest.fixture(autouse=True)
    def _restore(self):
        saved = litellm.enable_http2
        saved_cache = getattr(litellm, "in_memory_llm_clients_cache", None)
        yield
        litellm.enable_http2 = saved
        litellm.in_memory_llm_clients_cache = saved_cache

    @staticmethod
    def _pool_http2(client):
        return getattr(client._transport._pool, "_http2", None)

    def test_async_client_default_off_uses_aiohttp(self):
        litellm.enable_http2 = False
        client = BaseOpenAILLM._get_async_http_client()
        assert "Aiohttp" in type(client._transport).__name__

    def test_async_client_http2_enabled(self):
        litellm.enable_http2 = True
        client = BaseOpenAILLM._get_async_http_client()
        assert self._pool_http2(client) is True

    def test_async_shared_session_overrides_http2_and_warns(self):
        import asyncio

        from aiohttp import ClientSession

        litellm.enable_http2 = True

        async def _build():
            session = ClientSession()
            try:
                with patch(
                    "litellm.llms.openai.common_utils.verbose_logger"
                ) as mock_logger:
                    client = BaseOpenAILLM._get_async_http_client(
                        shared_session=session
                    )
                    return type(client._transport).__name__, mock_logger.warning.called
            finally:
                await session.close()

        tname, warned = asyncio.run(_build())
        assert "Aiohttp" in tname  # fell back to HTTP/1.1 transport
        assert warned  # and warned rather than silently downgrading

    def test_sync_client_http2_enabled_and_cached(self):
        litellm.enable_http2 = True
        litellm.in_memory_llm_clients_cache = None
        c1 = BaseOpenAILLM._get_sync_http_client()
        assert self._pool_http2(c1) is True
        # second call reuses the cached HTTPHandler client (shared pool)
        c2 = BaseOpenAILLM._get_sync_http_client()
        assert c1 is c2

    def test_sync_client_default_off_is_http1(self):
        litellm.enable_http2 = False
        litellm.in_memory_llm_clients_cache = None
        client = BaseOpenAILLM._get_sync_http_client()
        assert self._pool_http2(client) is False
