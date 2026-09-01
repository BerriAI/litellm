from unittest.mock import MagicMock, call, patch

import httpx
import openai
import pytest


import litellm
from litellm.litellm_core_utils.token_counter import token_counter
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


def test_evicting_a_client_built_on_the_callers_session_leaves_that_session_open(monkeypatch):
    """`litellm.aclient_session` belongs to the caller, who goes on using it.

    `_get_async_http_client` hands that session straight back, so the SDK client
    litellm builds around it is only a wrapper. The SDK's `close()` closes
    whatever http client it was given, so treating the wrapper as litellm's to
    close would close the caller's shared session out from under them.
    """
    import httpx

    from litellm.caching.evicted_client_closer import EvictedClientCloser
    from litellm.caching.llm_caching_handler import LLMClientCache
    from litellm.llms.openai.openai import OpenAIChatCompletion

    shared_session = httpx.AsyncClient()
    closer = EvictedClientCloser(grace_seconds=0.0)
    monkeypatch.setattr(litellm, "aclient_session", shared_session)
    monkeypatch.setattr(
        litellm,
        "in_memory_llm_clients_cache",
        LLMClientCache(evicted_client_closer=closer),
    )

    wrapper = OpenAIChatCompletion()._get_openai_client(
        is_async=True,
        api_key="sk-not-a-real-key",
        api_base="https://api.openai.com/v1",
        max_retries=2,
    )

    assert wrapper is not None
    assert wrapper._client is shared_session, "the wrapper should be built on the caller's session"

    closer.schedule(wrapper)
    closer.reap()

    assert closer.pending_count == 0, "a wrapper around the caller's session must never be queued"
    assert shared_session.is_closed is False, "closed the session the caller configured"


def test_a_client_litellm_built_its_own_http_client_for_is_still_closed(monkeypatch):
    """The ownership check must not turn the reclaim off for the ordinary case."""
    from litellm.caching.evicted_client_closer import EvictedClientCloser
    from litellm.caching.llm_caching_handler import LLMClientCache
    from litellm.llms.openai.openai import OpenAIChatCompletion

    closer = EvictedClientCloser(grace_seconds=0.0)
    monkeypatch.setattr(litellm, "aclient_session", None)
    monkeypatch.setattr(litellm, "client_session", None)
    monkeypatch.setattr(
        litellm,
        "in_memory_llm_clients_cache",
        LLMClientCache(evicted_client_closer=closer),
    )

    wrapper = OpenAIChatCompletion()._get_openai_client(
        is_async=False,
        api_key="sk-not-a-real-key",
        api_base="https://api.openai.com/v1",
        max_retries=2,
    )

    assert wrapper is not None
    closer.schedule(wrapper)

    assert closer.pending_count == 1, "litellm built this client's http client, so it owns it"

    closer.reap()

    assert wrapper.is_closed() is True


OUTPUT_LIMIT_400_MESSAGE = (
    "Could not finish the message because max_tokens or model output limit was reached. "
    "Please try again with higher max_tokens."
)
GENUINE_400_MESSAGE = "Invalid value for 'max_tokens': integer above maximum value. Expected <= 128000, got 999999999."
LONG_PROMPT = "please summarise the following notes for me: " + ("token " * 200)

CALL_KWARGS_BY_PROVIDER = {
    "openai": {"model": "gpt-5.6-sol", "api_key": "sk-not-a-real-key"},
    "azure": {
        "model": "azure/gpt-5.6-sol",
        "api_key": "not-a-real-key",
        "api_base": "https://not-a-real-resource.openai.azure.com",
        "api_version": "2024-10-21",
    },
}


def _transport(message: str) -> httpx.MockTransport:
    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": message, "type": "invalid_request_error"}})

    return httpx.MockTransport(_handler)


def _sync_client_raising(provider: str, message: str):
    http_client = httpx.Client(transport=_transport(message))
    if provider == "azure":
        return openai.AzureOpenAI(
            api_key="not-a-real-key",
            azure_endpoint="https://not-a-real-resource.openai.azure.com",
            api_version="2024-10-21",
            http_client=http_client,
        )
    return openai.OpenAI(api_key="sk-not-a-real-key", http_client=http_client)


def _async_client_raising(provider: str, message: str):
    http_client = httpx.AsyncClient(transport=_transport(message))
    if provider == "azure":
        return openai.AsyncAzureOpenAI(
            api_key="not-a-real-key",
            azure_endpoint="https://not-a-real-resource.openai.azure.com",
            api_version="2024-10-21",
            http_client=http_client,
        )
    return openai.AsyncOpenAI(api_key="sk-not-a-real-key", http_client=http_client)


def _completion_kwargs(provider: str, client, **overrides) -> dict:
    return {
        **CALL_KWARGS_BY_PROVIDER[provider],
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "client": client,
        **overrides,
    }


@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_sync_output_limit_400_maps_to_length_truncated_response(provider):
    response = litellm.completion(
        **_completion_kwargs(provider, _sync_client_raising(provider, OUTPUT_LIMIT_400_MESSAGE))
    )

    assert response.choices[0].finish_reason == "length"
    assert response.choices[0].message.content == ""
    assert response.usage.completion_tokens == 0


@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_mapped_response_still_bills_the_prompt_the_provider_processed(provider):
    messages = [{"role": "user", "content": LONG_PROMPT}]
    expected_prompt_tokens = token_counter(model="gpt-5.6-sol", messages=messages)
    assert expected_prompt_tokens > 100, "the fixture prompt must be big enough for a zeroed count to stand out"

    response = litellm.completion(
        **_completion_kwargs(provider, _sync_client_raising(provider, OUTPUT_LIMIT_400_MESSAGE), messages=messages)
    )

    assert response.usage.prompt_tokens == expected_prompt_tokens
    assert response.usage.completion_tokens == 0
    assert litellm.completion_cost(completion_response=response) > 0


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.asyncio
async def test_async_output_limit_400_maps_to_length_truncated_response(provider):
    response = await litellm.acompletion(
        **_completion_kwargs(provider, _async_client_raising(provider, OUTPUT_LIMIT_400_MESSAGE))
    )

    assert response.choices[0].finish_reason == "length"
    assert response.choices[0].message.content == ""
    assert response.usage.completion_tokens == 0


@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_sync_streaming_output_limit_400_maps_to_length_truncated_stream(provider):
    stream = litellm.completion(
        **_completion_kwargs(provider, _sync_client_raising(provider, OUTPUT_LIMIT_400_MESSAGE), stream=True)
    )
    chunks = list(stream)

    assert [c.choices[0].finish_reason for c in chunks].count("length") == 1
    assert all(not c.choices[0].delta.content for c in chunks)


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.asyncio
async def test_async_streaming_output_limit_400_maps_to_length_truncated_stream(provider):
    stream = await litellm.acompletion(
        **_completion_kwargs(provider, _async_client_raising(provider, OUTPUT_LIMIT_400_MESSAGE), stream=True)
    )
    chunks = [chunk async for chunk in stream]

    assert [c.choices[0].finish_reason for c in chunks].count("length") == 1
    assert all(not c.choices[0].delta.content for c in chunks)


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("stream", [False, True])
def test_sync_genuine_bad_request_still_raises(provider, stream):
    def _call_and_drain():
        result = litellm.completion(
            **_completion_kwargs(provider, _sync_client_raising(provider, GENUINE_400_MESSAGE), stream=stream)
        )
        list(result)

    with pytest.raises(litellm.BadRequestError):
        _call_and_drain()


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_async_genuine_bad_request_still_raises(provider, stream):
    async def _call_and_drain():
        result = await litellm.acompletion(
            **_completion_kwargs(provider, _async_client_raising(provider, GENUINE_400_MESSAGE), stream=stream)
        )
        async for _ in result:
            pass

    with pytest.raises(litellm.BadRequestError):
        await _call_and_drain()
