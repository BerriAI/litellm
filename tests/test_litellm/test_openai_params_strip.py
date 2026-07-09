import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest

import litellm
from litellm import acompletion, completion, embedding

litellm.return_response_headers = False


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
        try:
            await acompletion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                api_key="mock-key",
                litellm_params={"metadata": {"some_internal_key": "some_value"}},
                _litellm_test_param="test_value",
            )
        except Exception:
            pass

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
