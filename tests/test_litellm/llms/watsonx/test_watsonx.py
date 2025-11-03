import json
import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from typing import Optional
from unittest.mock import Mock, patch

import pytest

import litellm
from litellm import completion
from litellm.llms.custom_httpx.http_handler import HTTPHandler


@pytest.fixture
def watsonx_chat_completion_call():
    def _call(
        model="watsonx/my-test-model",
        messages=None,
        api_key="test_api_key",
        space_id: Optional[str] = None,
        headers=None,
        client=None,
        patch_token_call=True,
    ):
        if messages is None:
            messages = [{"role": "user", "content": "Hello, how are you?"}]
        if client is None:
            client = HTTPHandler()

        if patch_token_call:
            mock_response = Mock()
            mock_response.json.return_value = {
                "access_token": "mock_access_token",
                "expires_in": 3600,
            }
            mock_response.raise_for_status = Mock()  # No-op to simulate no exception

            with patch.object(client, "post") as mock_post, patch.object(
                litellm.module_level_client, "post", return_value=mock_response
            ) as mock_get:
                try:
                    completion(
                        model=model,
                        messages=messages,
                        api_key=api_key,
                        headers=headers or {},
                        client=client,
                        space_id=space_id,
                    )
                except Exception as e:
                    print(e)

                return mock_post, mock_get
        else:
            with patch.object(client, "post") as mock_post:
                try:
                    completion(
                        model=model,
                        messages=messages,
                        api_key=api_key,
                        headers=headers or {},
                        client=client,
                        space_id=space_id,
                    )
                except Exception as e:
                    print(e)
                return mock_post, None

    return _call


def test_watsonx_deployment_model_id_not_in_payload(
    monkeypatch, watsonx_chat_completion_call
):
    """Test that deployment models do not include 'model_id' in the request payload"""
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("WATSONX_API_BASE", "https://test-api.watsonx.ai")
    model = "watsonx/deployment/test-deployment-id"
    messages = [{"role": "user", "content": "Test message"}]

    mock_post, _ = watsonx_chat_completion_call(model=model, messages=messages)

    assert mock_post.call_count == 1
    json_data = json.loads(mock_post.call_args.kwargs["data"])
    # Ensure model_id is not in the payload for deployment models
    assert "model_id" not in json_data or json_data["model_id"] is None
    # Ensure project_id is also not in the payload for deployment models
    assert "project_id" not in json_data or json_data["project_id"] is None


def test_watsonx_regular_model_includes_model_id(
    monkeypatch, watsonx_chat_completion_call
):
    """Test that regular models include 'model_id' in the request payload"""
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("WATSONX_API_BASE", "https://test-api.watsonx.ai")
    model = "watsonx/regular-model"
    messages = [{"role": "user", "content": "Test message"}]

    mock_post, _ = watsonx_chat_completion_call(model=model, messages=messages)

    assert mock_post.call_count == 1
    json_data = json.loads(mock_post.call_args.kwargs["data"])
    # Ensure model_id is included in the payload for regular models
    assert "model_id" in json_data
    assert json_data["model_id"] == "regular-model"  # Provider prefix is stripped
    # Ensure project_id is also included for regular models
    assert "project_id" in json_data


@pytest.fixture
def watsonx_completion_call():
    def _call(
        model="watsonx_text/my-test-model",
        prompt="Hello, how are you?",
        api_key="test_api_key",
        space_id: Optional[str] = None,
        headers=None,
        client=None,
        patch_token_call=True,
    ):
        if client is None:
            client = HTTPHandler()

        if patch_token_call:
            mock_response = Mock()
            mock_response.json.return_value = {
                "access_token": "mock_access_token",
                "expires_in": 3600,
            }
            mock_response.raise_for_status = Mock()

            with patch.object(client, "post") as mock_post, patch.object(
                litellm.module_level_client, "post", return_value=mock_response
            ) as mock_get:
                try:
                    litellm.text_completion(
                        model=model,
                        prompt=prompt,
                        api_key=api_key,
                        headers=headers or {},
                        client=client,
                        space_id=space_id,
                    )
                except Exception as e:
                    print(e)

                return mock_post, mock_get
        else:
            with patch.object(client, "post") as mock_post:
                try:
                    litellm.text_completion(
                        model=model,
                        prompt=prompt,
                        api_key=api_key,
                        headers=headers or {},
                        client=client,
                        space_id=space_id,
                    )
                except Exception as e:
                    print(e)
                return mock_post, None

    return _call


def test_watsonx_completion_deployment_model_id_not_in_payload(
    monkeypatch, watsonx_completion_call
):
    """Test that deployment models do not include 'model_id' in completion request payload"""
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("WATSONX_API_BASE", "https://test-api.watsonx.ai")
    model = "watsonx_text/deployment/test-deployment-id"
    prompt = "Test prompt"

    mock_post, _ = watsonx_completion_call(model=model, prompt=prompt)

    assert mock_post.call_count == 1
    json_data = json.loads(mock_post.call_args.kwargs["data"])
    # Ensure model_id is not in the payload for deployment models
    assert "model_id" not in json_data
    # Ensure project_id is also not in the payload for deployment models
    assert "project_id" not in json_data


def test_watsonx_completion_regular_model_includes_model_id(
    monkeypatch, watsonx_completion_call
):
    """Test that regular models include 'model_id' in completion request payload"""
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("WATSONX_API_BASE", "https://test-api.watsonx.ai")
    model = "watsonx_text/regular-model"
    prompt = "Test prompt"

    mock_post, _ = watsonx_completion_call(model=model, prompt=prompt)

    assert mock_post.call_count == 1
    json_data = json.loads(mock_post.call_args.kwargs["data"])
    # Ensure model_id is included in the payload for regular models
    assert "model_id" in json_data
    assert json_data["model_id"] == "regular-model"  # Provider prefix is stripped
    # Ensure project_id is also included for regular models
    assert "project_id" in json_data


@pytest.mark.asyncio
async def test_watsonx_gpt_oss_prompt_transformation(monkeypatch):
    """
    Test that gpt-oss-120b model transforms messages to proper format instead of simple concatenation.
    
    This test starts from litellm.acompletion and verifies what gets sent in the final POST request body.
    Input messages should be transformed using the HuggingFace chat template from openai/gpt-oss-120b,
    not just concatenated as "You are chatgpt Hi there".
    """
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("WATSONX_API_BASE", "https://test-api.watsonx.ai")
    
    # Test with gpt-oss model using watsonx_text provider (text generation endpoint)
    model = "watsonx_text/openai/gpt-oss-120b"
    
    # Input messages
    messages = [
        {"role": "system", "content": "You are chatgpt"},
        {"role": "user", "content": "Hi there"}
    ]
    
    # Mock the HTTP client
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    
    client = AsyncHTTPHandler()
    
    # Mock the token call
    mock_token_response = Mock()
    mock_token_response.json.return_value = {
        "access_token": "mock_access_token",
        "expires_in": 3600,
    }
    mock_token_response.raise_for_status = Mock()
    
    # Mock the completion call
    mock_completion_response = Mock()
    mock_completion_response.status_code = 200
    mock_completion_response.json.return_value = {
        "results": [{
            "generated_text": "Hello! How can I help you?",
            "generated_token_count": 10,
            "input_token_count": 5
        }],
        "model_id": "openai/gpt-oss-120b"
    }
    
    with patch.object(client, "post") as mock_post, patch.object(
        litellm.module_level_client, "post", return_value=mock_token_response
    ):
        # Set the mock to return the completion response
        mock_post.return_value = mock_completion_response
        
        try:
            # Call acompletion with messages
            await litellm.acompletion(
                model=model,
                messages=messages,
                api_key="test_api_key",
                client=client,
            )
        except Exception as e:
            # May fail due to incomplete mocking, but we should have captured the request
            print(f"Exception (may be expected): {e}")
    
    # Verify the POST was called
    assert mock_post.call_count >= 1, f"POST should have been called at least once, got {mock_post.call_count}"
    
    # Get the request body from the first call
    call_args = mock_post.call_args
    json_data = json.loads(call_args.kwargs["data"])
    
    print(f"\n{'='*80}")
    print(f"Input messages to litellm.acompletion:")
    print(json.dumps(messages, indent=2))
    print(f"\n{'='*80}")
    print(f"Final POST request body:")
    print(json.dumps(json_data, indent=2))
    print(f"{'='*80}\n")
    
    # Verify the transformed input is in the request
    assert "input" in json_data, "Request should have 'input' field"
    transformed_prompt = json_data["input"]
    
    print(f"Transformed prompt: {repr(transformed_prompt)}")
    print(f"Prompt length: {len(transformed_prompt)}")
    
    # Verify it's NOT simple concatenation
    simple_concat = "You are chatgpt Hi there"
    assert transformed_prompt != simple_concat, (
        f"Prompt should not be simple concatenation.\n"
        f"Expected: Chat template with <|start|> tags\n"
        f"Got: {transformed_prompt}"
    )
    
    # Verify it contains proper chat template formatting
    assert "<|start|>" in transformed_prompt, "Prompt should contain <|start|> tag"
    assert "<|message|>" in transformed_prompt, "Prompt should contain <|message|> tag"
    assert "<|end|>" in transformed_prompt, "Prompt should contain <|end|> tag"
    assert "You are chatgpt" in transformed_prompt, "Prompt should contain system message content"
    assert "Hi there" in transformed_prompt, "Prompt should contain user message content"


@pytest.mark.asyncio
async def test_watsonx_gpt_oss_uses_async_http_handler():
    """
    Test that verifies async HTTP client is used when fetching HuggingFace templates.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.litellm_core_utils.prompt_templates.huggingface_template_handler import (
        _aget_chat_template_file,
    )

    # Mock the async HTTP client
    mock_async_client = MagicMock()
    mock_get = AsyncMock()
    mock_async_client.get = mock_get
    
    # Create mock response for chat template file
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"test template content"
    mock_get.return_value = mock_response
    
    # Test the async function directly
    with patch("litellm.litellm_core_utils.prompt_templates.huggingface_template_handler.get_async_httpx_client", return_value=mock_async_client):
        result = await _aget_chat_template_file(hf_model_name="test/model")
        
        # Verify async HTTP client was called
        assert mock_get.called, "Async HTTP client's get method should be called"
        assert mock_get.await_count > 0, "Async HTTP client's get should be awaited"
        
        # Verify it was called with HuggingFace URL
        call_args = mock_get.call_args
        assert call_args is not None, "get should have been called with arguments"
        called_url = call_args.kwargs.get("url", "")
        assert "huggingface.co/test/model" in called_url, f"Should call HuggingFace API for test/model, got: {called_url}"
        assert result["status"] == "success", "Should return success status"
