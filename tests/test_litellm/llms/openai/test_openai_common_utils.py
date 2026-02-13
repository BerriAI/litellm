import inspect
import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.openai.common_utils import BaseOpenAILLM, _CACHE_KEY_PARAMS

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
    with patch(client_path) as mock_client_class, patch.object(
        BaseOpenAILLM, "set_cached_openai_client"
    ) as mock_set_cache, patch.object(
        BaseOpenAILLM, "get_cached_openai_client"
    ) as mock_get_cache:
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

    expected_openai = [
        p for p in inspect.signature(OpenAI.__init__).parameters if p != "self"
    ]
    expected_azure = [
        p for p in inspect.signature(AzureOpenAI.__init__).parameters if p != "self"
    ]

    assert _OPENAI_INIT_PARAMS == expected_openai
    assert _AZURE_OPENAI_INIT_PARAMS == expected_azure


@pytest.mark.parametrize("client_type", ["openai", "azure"])
def test_get_openai_client_initialization_param_fields(client_type):
    """Verify the method returns the correct pre-computed params for each client type."""
    result = BaseOpenAILLM.get_openai_client_initialization_param_fields(client_type)
    assert isinstance(result, list)
    assert len(result) > 0
    assert "self" not in result


# --- Safety-net tests for cache key coverage ---
# _CACHE_KEY_PARAMS is imported from common_utils (single source of truth)

# _get_openai_client params that don't affect client identity
_IGNORED_OPENAI = {"self", "client", "shared_session"}

# get_azure_openai_client params that don't affect client identity
# _is_async maps to is_async in the cache key dict
_IGNORED_AZURE = {"self", "client", "litellm_params", "model"}
_AZURE_CACHE_KEY_PARAMS = _CACHE_KEY_PARAMS | {"_is_async"}


def test_openai_cache_key_covers_all_params():
    """
    If someone adds a new param to _get_openai_client that affects client
    identity, this test will fail — signaling the cache key needs updating.
    """
    from litellm.llms.openai.openai import OpenAIChatCompletion

    sig_params = set(
        inspect.signature(OpenAIChatCompletion._get_openai_client).parameters.keys()
    )
    covered = _CACHE_KEY_PARAMS | _IGNORED_OPENAI
    uncovered = sig_params - covered
    assert uncovered == set(), (
        f"_get_openai_client has new param(s) {uncovered} not in cache key or ignore list. "
        f"Update get_openai_client_cache_key or add to _IGNORED_OPENAI."
    )


def test_azure_cache_key_covers_all_params():
    """
    If someone adds a new param to get_azure_openai_client that affects client
    identity, this test will fail — signaling the cache key needs updating.
    """
    from litellm.llms.azure.common_utils import BaseAzureLLM

    sig_params = set(
        inspect.signature(BaseAzureLLM.get_azure_openai_client).parameters.keys()
    )
    covered = _AZURE_CACHE_KEY_PARAMS | _IGNORED_AZURE
    uncovered = sig_params - covered
    assert uncovered == set(), (
        f"get_azure_openai_client has new param(s) {uncovered} not in cache key or ignore list. "
        f"Update get_openai_client_cache_key or add to _IGNORED_AZURE."
    )


def test_cache_key_format():
    """
    Verify cache key contains all expected components and that different
    api_version values produce different keys (Azure collision regression).
    """
    import hashlib

    params = {
        "api_key": "sk-test-key-123",
        "is_async": True,
        "api_base": "https://api.openai.com/v1",
        "api_version": "2024-02-01",
        "timeout": 600,
        "max_retries": 2,
        "organization": "org-abc",
    }

    key = BaseOpenAILLM.get_openai_client_cache_key(params, "openai")

    expected_hash = hashlib.sha256(b"sk-test-key-123").hexdigest()
    assert "openai" in key
    assert expected_hash in key
    assert "True" in key  # is_async
    assert "https://api.openai.com/v1" in key
    assert "2024-02-01" in key
    assert "600" in key
    assert "2" in key  # max_retries
    assert "org-abc" in key

    # Azure collision regression: different api_version must produce different keys
    params_v1 = {**params, "api_version": "2024-02-01"}
    params_v2 = {**params, "api_version": "2024-06-01"}
    key_v1 = BaseOpenAILLM.get_openai_client_cache_key(params_v1, "azure")
    key_v2 = BaseOpenAILLM.get_openai_client_cache_key(params_v2, "azure")
    assert key_v1 != key_v2, "Different api_version values must produce different cache keys"
