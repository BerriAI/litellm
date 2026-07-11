import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest

import litellm
from litellm import acompletion, completion, embedding

litellm.return_response_headers = False


@pytest.fixture(autouse=True)
def clear_client_cache():
    """
    Clear the HTTP client cache before each test to ensure mocks are used.
    This prevents cached real clients from being reused across tests.
    """
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is not None:
        cache.flush_cache()
    yield
    if cache is not None:
        cache.flush_cache()


@pytest.mark.asyncio
async def test_openai_chat_completion_params_strip():
    """
    Test that litellm_params and _litellm_* prefixed params are stripped
    from OpenAI completion calls.
    """
    # Mock return value of parse() which is what is called on raw_response
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.index = 0
    mock_choice.message = MagicMock(content="Mock response", role="assistant")
    mock_choice.message.tool_calls = None
    mock_choice.message.function_call = None
    mock_choice.message.provider_specific_fields = {}

    mock_response_data = MagicMock()
    mock_response_data.choices = [mock_choice]
    mock_response_data.id = "chatcmpl-123"
    mock_response_data.created = 1677858242
    mock_response_data.model = "gpt-4o"
    mock_response_data.object = "chat.completion"
    mock_response_data.usage = MagicMock(completion_tokens=10, prompt_tokens=5, total_tokens=15)

    # We mock the underlying client create call
    mock_create = MagicMock()
    mock_raw_resp = MagicMock()
    mock_raw_resp.headers = {"x-test-header": "test"}
    mock_raw_resp.parse.return_value = mock_response_data
    mock_create.return_value = mock_raw_resp

    with patch("openai.resources.chat.completions.Completions.create", mock_create):
        completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            api_key="mock-key",
            # internal params that should be stripped
            litellm_params={"metadata": {"some_internal_key": "some_value"}},
            _litellm_test_param="test_value",
        )

        # Verify call arguments
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]

        # Verify that internal params are not in the top-level keys or extra_body
        assert "litellm_params" not in call_kwargs
        assert "_litellm_test_param" not in call_kwargs

        extra_body = call_kwargs.get("extra_body", {})
        if extra_body:
            assert "litellm_params" not in extra_body
            assert "_litellm_test_param" not in extra_body


@pytest.mark.asyncio
async def test_openai_chat_acompletion_params_strip():
    """
    Test that litellm_params and _litellm_* prefixed params are stripped
    from OpenAI async completion calls.
    """
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.index = 0
    mock_choice.message = MagicMock(content="Mock response", role="assistant")
    mock_choice.message.tool_calls = None
    mock_choice.message.function_call = None
    mock_choice.message.provider_specific_fields = {}

    mock_response_data = MagicMock()
    mock_response_data.choices = [mock_choice]
    mock_response_data.id = "chatcmpl-123"
    mock_response_data.created = 1677858242
    mock_response_data.model = "gpt-4o"
    mock_response_data.object = "chat.completion"
    mock_response_data.usage = MagicMock(completion_tokens=10, prompt_tokens=5, total_tokens=15)

    mock_raw_resp = MagicMock()
    mock_raw_resp.headers = {"x-test-header": "test"}
    mock_raw_resp.parse.return_value = mock_response_data

    mock_acreate = AsyncMock(return_value=mock_raw_resp)

    with patch("openai.resources.chat.completions.AsyncCompletions.create", mock_acreate):
        await acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            api_key="mock-key",
            litellm_params={"metadata": {"some_internal_key": "some_value"}},
            _litellm_test_param="test_value",
        )

        mock_acreate.assert_called_once()
        call_kwargs = mock_acreate.call_args[1]

        assert "litellm_params" not in call_kwargs
        assert "_litellm_test_param" not in call_kwargs

        extra_body = call_kwargs.get("extra_body", {})
        if extra_body:
            assert "litellm_params" not in extra_body
            assert "_litellm_test_param" not in extra_body


@pytest.mark.asyncio
async def test_openai_embedding_params_strip():
    """
    Test that litellm_params and _litellm_* prefixed params are stripped
    from OpenAI embedding calls.
    """
    mock_response_data = MagicMock()
    mock_response_data.model = "text-embedding-3-small"
    mock_response_data.object = "list"
    mock_response_data.data = [MagicMock(embedding=[0.1, 0.2])]
    mock_response_data.usage = MagicMock(prompt_tokens=5, total_tokens=5)

    mock_create = MagicMock()
    mock_raw_resp = MagicMock()
    mock_raw_resp.headers = {"x-test-header": "test"}
    mock_raw_resp.parse.return_value = mock_response_data
    mock_create.return_value = mock_raw_resp

    with patch("openai.resources.embeddings.Embeddings.create", mock_create):
        embedding(
            model="text-embedding-3-small",
            input=["hello"],
            api_key="mock-key",
            litellm_params={"metadata": {"some_internal_key": "some_value"}},
            _litellm_test_param="test_value",
        )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]

        assert "litellm_params" not in call_kwargs
        assert "_litellm_test_param" not in call_kwargs

        extra_body = call_kwargs.get("extra_body", {})
        if extra_body:
            assert "litellm_params" not in extra_body
            assert "_litellm_test_param" not in extra_body


@pytest.mark.asyncio
async def test_openai_metadata_preview_feature():
    """
    Test that when litellm.enable_preview_features = True,
    metadata is successfully forwarded and not stripped.
    And when it is False, metadata is not in the request.
    """
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.index = 0
    mock_choice.message = MagicMock(content="Mock response", role="assistant")
    mock_choice.message.tool_calls = None
    mock_choice.message.function_call = None
    mock_choice.message.provider_specific_fields = {}

    mock_response_data = MagicMock()
    mock_response_data.choices = [mock_choice]
    mock_response_data.id = "chatcmpl-123"
    mock_response_data.created = 1677858242
    mock_response_data.model = "gpt-4o"
    mock_response_data.object = "chat.completion"
    mock_response_data.usage = MagicMock(completion_tokens=10, prompt_tokens=5, total_tokens=15)

    mock_raw_resp = MagicMock()
    mock_raw_resp.headers = {"x-test-header": "test"}
    mock_raw_resp.parse.return_value = mock_response_data

    # 1. Test with enable_preview_features = True (metadata should be forwarded)
    mock_create_preview = MagicMock(return_value=mock_raw_resp)
    litellm.enable_preview_features = True
    try:
        with patch("openai.resources.chat.completions.Completions.create", mock_create_preview):
            completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                api_key="mock-key",
                metadata={"user_api_key_user_id": "test_user_id"},
            )
            mock_create_preview.assert_called_once()
            call_kwargs = mock_create_preview.call_args[1]
            assert "metadata" in call_kwargs
            assert call_kwargs["metadata"] == {"user_api_key_user_id": "test_user_id"}
    finally:
        litellm.enable_preview_features = False
        cache = getattr(litellm, "in_memory_llm_clients_cache", None)
        if cache is not None:
            cache.flush_cache()

    # 2. Test with enable_preview_features = False (metadata should be stripped/omitted)
    mock_create_no_preview = MagicMock(return_value=mock_raw_resp)
    with patch("openai.resources.chat.completions.Completions.create", mock_create_no_preview):
        completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            api_key="mock-key",
            metadata={"user_api_key_user_id": "test_user_id"},
        )
        mock_create_no_preview.assert_called_once()
        call_kwargs = mock_create_no_preview.call_args[1]
        assert "metadata" not in call_kwargs
