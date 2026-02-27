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
        {"role": "user", "content": "Hi there"},
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
        "results": [
            {
                "generated_text": "Hello! How can I help you?",
                "generated_token_count": 10,
                "input_token_count": 5,
                "stop_reason": "stop",  # Required field for response transformation
            }
        ],
        "model_id": "openai/gpt-oss-120b",
    }

    # Mock HuggingFace template fetch to make test deterministic and avoid network flakiness.
    # The test verifies that prompt transformation occurs (not simple concatenation), not the exact
    # HuggingFace template format. Using a mock template that produces the correct format is sufficient.
    from unittest.mock import patch

    # Mock template that produces gpt-oss-120b-like format.
    # Note: This is a simplified version of the actual template. The real template is more complex
    # (adds metadata, handles tools, thinking messages, etc.), but this captures the key aspects:
    # - Converts system role to developer (matching real template behavior)
    # - Uses the same tag structure (<|start|>, <|message|>, <|end|>)
    # - Preserves message content
    mock_tokenizer_config = {
        "status": "success",
        "tokenizer": {
            "chat_template": "{% for message in messages %}{% if message['role'] == 'system' %}<|start|>developer<|message|>{% else %}<|start|>{{ message['role'] }}<|message|>{% endif %}{{ message['content'] }}<|end|>{% endfor %}",
            "bos_token": None,
            "eos_token": None,
        },
    }

    async def mock_aget_tokenizer_config(hf_model_name: str):
        return mock_tokenizer_config

    async def mock_aget_chat_template_file(hf_model_name: str):
        # Return failure to use tokenizer_config instead
        return {"status": "failure"}

    # Set cached tokenizer config directly to avoid race conditions with parallel tests.
    # When running with pytest-xdist (-n 16), another test might populate the cache between
    # clearing it and the actual usage. By setting the cache directly, we ensure the correct
    # template is always used regardless of test execution order.
    hf_model = "openai/gpt-oss-120b"
    litellm.known_tokenizer_config[hf_model] = mock_tokenizer_config

    # Also create sync mock functions in case the fallback sync path is used
    def mock_get_tokenizer_config(hf_model_name: str):
        return mock_tokenizer_config

    def mock_get_chat_template_file(hf_model_name: str):
        return {"status": "failure"}

    # Async mock function for client.post to properly handle async method mocking
    async def mock_post_func(*args, **kwargs):
        return mock_completion_response

    # Mock the token generation response to avoid actual API call
    mock_token_get_response = Mock()
    mock_token_get_response.json.return_value = {
        "access_token": "mock_access_token",
        "expires_in": 3600,
    }
    mock_token_get_response.raise_for_status = Mock()

    with patch.object(client, "post", side_effect=mock_post_func) as mock_post, patch.object(
        litellm.module_level_client, "post", return_value=mock_token_get_response
    ), patch(
        "litellm.litellm_core_utils.prompt_templates.huggingface_template_handler._aget_tokenizer_config",
        side_effect=mock_aget_tokenizer_config,
    ), patch(
        "litellm.litellm_core_utils.prompt_templates.huggingface_template_handler._aget_chat_template_file",
        side_effect=mock_aget_chat_template_file,
    ), patch(
        "litellm.litellm_core_utils.prompt_templates.huggingface_template_handler._get_tokenizer_config",
        side_effect=mock_get_tokenizer_config,
    ), patch(
        "litellm.litellm_core_utils.prompt_templates.huggingface_template_handler._get_chat_template_file",
        side_effect=mock_get_chat_template_file,
    ):
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
    assert (
        mock_post.call_count >= 1
    ), f"POST should have been called at least once, got {mock_post.call_count}"

    # Get the request body from the first call
    # Use call_args_list to be more robust - get the first call's arguments
    assert len(mock_post.call_args_list) > 0, "mock_post should have at least one call"
    call_args = mock_post.call_args_list[0]
    assert call_args is not None, "call_args should not be None"
    assert "data" in call_args.kwargs, "call_args.kwargs should contain 'data'"
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

    # Verify transformation occurred
    assert transformed_prompt is not None, (
        "Prompt transformation failed - the template should have been applied to transform "
        "messages into the correct format for gpt-oss-120b."
    )

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
    assert (
        "You are chatgpt" in transformed_prompt
    ), "Prompt should contain system message content"
    assert (
        "Hi there" in transformed_prompt
    ), "Prompt should contain user message content"


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
    with patch(
        "litellm.litellm_core_utils.prompt_templates.huggingface_template_handler.get_async_httpx_client",
        return_value=mock_async_client,
    ):
        result = await _aget_chat_template_file(hf_model_name="test/model")

        # Verify async HTTP client was called
        assert mock_get.called, "Async HTTP client's get method should be called"
        assert mock_get.await_count > 0, "Async HTTP client's get should be awaited"

        # Verify it was called with HuggingFace URL
        call_args = mock_get.call_args
        assert call_args is not None, "get should have been called with arguments"
        called_url = call_args.kwargs.get("url", "")
        assert (
            "huggingface.co/test/model" in called_url
        ), f"Should call HuggingFace API for test/model, got: {called_url}"
        assert result["status"] == "success", "Should return success status"


def test_watsonx_chat_completion_with_reasoning_effort(monkeypatch):
    """
    Test that 'reasoning_effort' is correctly passed through to the WatsonX API payload.
    """
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("WATSONX_API_BASE", "https://test-api.watsonx.ai")

    model = "watsonx/openai/gpt-oss-120b"
    messages = [{"role": "user", "content": "Test message"}]

    client = HTTPHandler()

    # Mock the token generation call
    mock_token_response = Mock()
    mock_token_response.json.return_value = {
        "access_token": "mock_access_token",
        "expires_in": 3600,
    }
    mock_token_response.raise_for_status = Mock()

    # Call litellm.completion with the new parameter
    with patch.object(client, "post") as mock_post, patch.object(
        litellm.module_level_client, "post", return_value=mock_token_response
    ):
        try:
            completion(
                model=model,
                messages=messages,
                api_key="test_api_key",
                client=client,
                reasoning_effort="low",
            )
        except Exception as e:
            print(f"Caught expected exception: {e}")

    # Verify the parameter is in the final request payload
    assert (
        mock_post.call_count == 1
    ), "The completion endpoint should have been called once."

    # Get the JSON data sent in the POST request
    request_kwargs = mock_post.call_args.kwargs
    json_data = json.loads(request_kwargs["data"])

    print("\nRequest payload sent to WatsonX API:")
    print(json.dumps(json_data, indent=2))

    # Check for the parameter at the top level of the payload
    assert (
        "reasoning_effort" in json_data
    ), "'reasoning_effort' should be at the top level of the payload."
    assert (
        json_data["reasoning_effort"] == "low"
    ), "The value of 'reasoning_effort' should be 'low'."


def test_watsonx_zen_api_key_from_client(monkeypatch, watsonx_chat_completion_call):
    """
    Test that zen_api_key can be passed from client code and is used in Authorization header.
    """
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("WATSONX_API_BASE", "https://test-api.watsonx.ai")

    model = "watsonx/ibm/granite-3-3-8b-instruct"
    messages = [{"role": "user", "content": "What is your favorite color?"}]

    client = HTTPHandler()

    zen_api_key = "U1ZDLWQo="

    # No need to patch token call since zen_api_key should skip token generation
    with patch.object(client, "post") as mock_post:
        try:
            completion(
                model=model,
                messages=messages,
                api_key="test_api_key",
                client=client,
                zen_api_key=zen_api_key,
            )
        except Exception as e:
            print(f"Caught expected exception: {e}")

    # Verify the request was made
    assert mock_post.call_count == 1, "The completion endpoint should have been called once."

    # Get the headers sent in the POST request
    request_kwargs = mock_post.call_args.kwargs
    headers = request_kwargs["headers"]

    print("\nHeaders sent to WatsonX API:")
    print(json.dumps(dict(headers), indent=2))

    # Verify Authorization header uses ZenApiKey format
    assert "Authorization" in headers, "Authorization header should be present."
    assert headers["Authorization"] == f"ZenApiKey {zen_api_key}", (
        f"Authorization header should use ZenApiKey format. "
        f"Expected: 'ZenApiKey {zen_api_key}', Got: '{headers['Authorization']}'"
    )


def test_watsonx_zen_api_key_from_env(monkeypatch, watsonx_chat_completion_call):
    """
    Test that zen_api_key from environment variable is used in Authorization header.
    """
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("WATSONX_API_BASE", "https://test-api.watsonx.ai")

    zen_api_key = "U1ZDLWxpdG--==="
    monkeypatch.setenv("WATSONX_ZENAPIKEY", zen_api_key)

    model = "watsonx/ibm/granite-3-3-8b-instruct"
    messages = [{"role": "user", "content": "What is your favorite color?"}]

    client = HTTPHandler()

    # No need to patch token call since zen_api_key should skip token generation
    with patch.object(client, "post") as mock_post:
        try:
            completion(
                model=model,
                messages=messages,
                api_key="test_api_key",
                client=client,
            )
        except Exception as e:
            print(f"Caught expected exception: {e}")

    # Verify the request was made
    assert mock_post.call_count == 1, "The completion endpoint should have been called once."

    # Get the headers sent in the POST request
    request_kwargs = mock_post.call_args.kwargs
    headers = request_kwargs["headers"]

    print("\nHeaders sent to WatsonX API:")
    print(json.dumps(dict(headers), indent=2))

    # Verify Authorization header uses ZenApiKey format
    assert "Authorization" in headers, "Authorization header should be present."
    assert headers["Authorization"] == f"ZenApiKey {zen_api_key}", (
        f"Authorization header should use ZenApiKey format. "
        f"Expected: 'ZenApiKey {zen_api_key}', Got: '{headers['Authorization']}'"
    )
