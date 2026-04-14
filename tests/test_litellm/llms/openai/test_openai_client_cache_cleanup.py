import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.llms.openai.openai import OpenAIChatCompletion
from openai import AsyncOpenAI, OpenAI


@pytest.fixture(autouse=True)
def _clear_in_memory_cache():
    """Clear global cache to avoid cross-test contamination."""
    litellm.in_memory_llm_clients_cache.cache_dict.clear()
    yield
    litellm.in_memory_llm_clients_cache.cache_dict.clear()


class TestHttpClientClosedHandling:
    def test_httpx_is_closed_returns_true_when_client_is_none(self):
        chat = OpenAIChatCompletion()
        assert chat._httpx_is_closed(None) is True

    def test_httpx_is_closed_returns_true_when_client_attribute_missing(self):
        chat = OpenAIChatCompletion()
        client_without__client = SimpleNamespace()  # no _client attribute
        assert chat._httpx_is_closed(client_without__client) is True

    def test_httpx_is_closed_returns_true_when_httpx_is_closed(self):
        chat = OpenAIChatCompletion()
        openai_client = SimpleNamespace(_client=SimpleNamespace(is_closed=True))
        assert chat._httpx_is_closed(openai_client) is True

    def test_httpx_is_closed_returns_false_when_httpx_is_open(self):
        chat = OpenAIChatCompletion()
        openai_client = SimpleNamespace(_client=SimpleNamespace(is_closed=False))
        assert chat._httpx_is_closed(openai_client) is False


class TestOpenAIClientCacheLogic:
    @patch.object(OpenAIChatCompletion, "_set_dynamic_params_on_client")
    def test_explicit_client_bypasses_cache(self, mock_set_dynamic):
        chat = OpenAIChatCompletion()
        explicit_client = MagicMock(spec=OpenAI)

        with patch.object(
            OpenAIChatCompletion, "get_cached_openai_client"
        ) as mock_get_cache:
            result = chat._get_openai_client(
                is_async=False,
                api_key="test-key",
                client=explicit_client,
                max_retries=3,
            )

        assert result is explicit_client
        mock_get_cache.assert_not_called()
        mock_set_dynamic.assert_called_once_with(
            client=explicit_client, organization=None, max_retries=3
        )

    def test_max_retries_must_be_int_raises(self):
        chat = OpenAIChatCompletion()
        with pytest.raises(Exception) as excinfo:
            chat._get_openai_client(
                is_async=False,
                api_key="test-key",
                max_retries="3",  # not int
            )
        assert "max retries must be an int" in str(excinfo.value)

    @patch.object(OpenAIChatCompletion, "set_cached_openai_client")
    @patch.object(OpenAIChatCompletion, "get_cached_openai_client")
    def test_cached_async_client_open_is_returned(self, mock_get_cache, mock_set_cache):
        chat = OpenAIChatCompletion()

        cached_async = object.__new__(AsyncOpenAI)
        cached_async._client = SimpleNamespace(is_closed=False)
        mock_get_cache.return_value = cached_async

        with patch.object(
            AsyncOpenAI, "__init__", return_value=None
        ) as mock_async_init:
            result = chat._get_openai_client(
                is_async=True,
                api_key="test-key",
                max_retries=3,
            )

        assert result is cached_async
        mock_async_init.assert_not_called()
        mock_set_cache.assert_not_called()

    @patch.object(OpenAIChatCompletion, "set_cached_openai_client")
    @patch.object(OpenAIChatCompletion, "get_cached_openai_client")
    def test_cached_async_client_closed_is_evicted_and_new_created(
        self, mock_get_cache, mock_set_cache
    ):
        chat = OpenAIChatCompletion()

        cached_async = object.__new__(AsyncOpenAI)
        cached_async._client = SimpleNamespace(is_closed=True)
        mock_get_cache.return_value = cached_async

        # Store it under the *event-loop-suffixed* key, like set_cache() does
        raw_key = "fake-cache-key"
        loop_key = litellm.in_memory_llm_clients_cache.update_cache_key_with_event_loop(
            raw_key
        )
        litellm.in_memory_llm_clients_cache.cache_dict[loop_key] = cached_async

        with patch(
            "litellm.llms.openai.openai.BaseOpenAILLM.get_openai_client_cache_key",
            return_value=raw_key,
        ), patch.object(
            OpenAIChatCompletion, "_get_async_http_client", return_value=MagicMock()
        ), patch.object(
            AsyncOpenAI, "__init__", return_value=None
        ) as mock_async_init, patch.object(
            litellm.in_memory_llm_clients_cache,
            "_remove_key",
            wraps=litellm.in_memory_llm_clients_cache._remove_key,
        ) as mock_remove_key:
            result = chat._get_openai_client(
                is_async=True,
                api_key="test-key",
                max_retries=3,
            )

        assert isinstance(result, AsyncOpenAI)
        assert result is not cached_async

        # Evicted using the loop-suffixed key
        assert loop_key not in litellm.in_memory_llm_clients_cache.cache_dict
        mock_remove_key.assert_called_once_with(loop_key)

        mock_set_cache.assert_called_once()
        mock_async_init.assert_called_once()

    @patch.object(OpenAIChatCompletion, "set_cached_openai_client")
    @patch.object(OpenAIChatCompletion, "get_cached_openai_client")
    def test_cached_sync_client_open_is_returned(self, mock_get_cache, mock_set_cache):
        chat = OpenAIChatCompletion()

        cached_sync = object.__new__(OpenAI)
        cached_sync._client = SimpleNamespace(is_closed=False)
        mock_get_cache.return_value = cached_sync

        with patch.object(OpenAI, "__init__", return_value=None) as mock_sync_init:
            result = chat._get_openai_client(
                is_async=False,
                api_key="test-key",
                max_retries=3,
            )

        assert result is cached_sync
        mock_sync_init.assert_not_called()
        mock_set_cache.assert_not_called()

    @patch.object(OpenAIChatCompletion, "set_cached_openai_client")
    @patch.object(OpenAIChatCompletion, "get_cached_openai_client")
    def test_cached_sync_client_closed_is_evicted_and_new_created(
        self, mock_get_cache, mock_set_cache
    ):
        chat = OpenAIChatCompletion()

        cached_sync = object.__new__(OpenAI)
        cached_sync._client = SimpleNamespace(is_closed=True)
        mock_get_cache.return_value = cached_sync

        raw_key = "fake-cache-key"
        loop_key = litellm.in_memory_llm_clients_cache.update_cache_key_with_event_loop(
            raw_key
        )
        litellm.in_memory_llm_clients_cache.cache_dict[loop_key] = cached_sync

        with patch(
            "litellm.llms.openai.openai.BaseOpenAILLM.get_openai_client_cache_key",
            return_value=raw_key,
        ), patch.object(
            OpenAIChatCompletion, "_get_sync_http_client", return_value=MagicMock()
        ), patch.object(
            OpenAI, "__init__", return_value=None
        ) as mock_sync_init, patch.object(
            litellm.in_memory_llm_clients_cache,
            "_remove_key",
            wraps=litellm.in_memory_llm_clients_cache._remove_key,
        ) as mock_remove_key:
            result = chat._get_openai_client(
                is_async=False,
                api_key="test-key",
                max_retries=3,
            )

        assert isinstance(result, OpenAI)
        assert result is not cached_sync

        assert loop_key not in litellm.in_memory_llm_clients_cache.cache_dict
        mock_remove_key.assert_called_once_with(loop_key)

        mock_set_cache.assert_called_once()
        mock_sync_init.assert_called_once()
