from typing import Optional
from unittest.mock import patch

import pytest

import litellm
from litellm.llms.llamafile.chat.transformation import LlamafileChatConfig


@pytest.mark.parametrize(
    "input_api_key, env_api_key, expected_api_key",
    [
        ("user-provided-key", "secret-key", "user-provided-key"),
        (None, "secret-key", "secret-key"),
        (None, None, "fake-api-key"),
        ("", "secret-key", "secret-key"),  # Empty string should fall back to secret
        ("", None, "fake-api-key"),  # Empty string with no secret should use the fake key
    ],
)
def test_resolve_api_key(
    input_api_key, env_api_key, expected_api_key
):
    env = {}
    if env_api_key is not None:
        env["LLAMAFILE_API_KEY"] = env_api_key
        
    with patch.dict("os.environ", env, clear=True):
        result = LlamafileChatConfig._resolve_api_key(input_api_key)
        assert result == expected_api_key


@pytest.mark.parametrize(
    "input_api_base, env_api_base, expected_api_base",
    [
        (
            "https://user-api.example.com",
            "https://secret-api.example.com",
            "https://user-api.example.com",
        ),
        (
            None,
            "https://secret-api.example.com",
            "https://secret-api.example.com",
        ),
        (None, None, "http://127.0.0.1:8080/v1"),
        (
            "",
            "https://secret-api.example.com",
            "https://secret-api.example.com",
        ),  # Empty string should fall back
    ],
)
def test_resolve_api_base(
    input_api_base,
    env_api_base,
    expected_api_base,
):
    env = {}
    if env_api_base is not None:
        env["LLAMAFILE_API_BASE"] = env_api_base
        
    with patch.dict("os.environ", env, clear=True):
        result = LlamafileChatConfig._resolve_api_base(input_api_base)
        assert result == expected_api_base


@pytest.mark.parametrize(
    "api_base, api_key, env_base, env_key, expected_base, expected_key",
    [
        # User-provided values
        (
            "https://user-api.example.com",
            "user-key",
            "https://secret-api.example.com",
            "secret-key",
            "https://user-api.example.com",
            "user-key",
        ),
        # Fallback to env vars
        (
            None,
            None,
            "https://secret-api.example.com",
            "secret-key",
            "https://secret-api.example.com",
            "secret-key",
        ),
        # Nothing provided, use defaults
        (None, None, None, None, "http://127.0.0.1:8080/v1", "fake-api-key"),
        # Mixed scenarios
        (
            "https://user-api.example.com",
            None,
            None,
            "secret-key",
            "https://user-api.example.com",
            "secret-key",
        ),
        (
            None,
            "user-key",
            "https://secret-api.example.com",
            None,
            "https://secret-api.example.com",
            "user-key",
        ),
    ],
)
def test_get_openai_compatible_provider_info(
    api_base, api_key, env_base, env_key, expected_base, expected_key
):
    config = LlamafileChatConfig()
    
    env = {}
    if env_base is not None:
        env["LLAMAFILE_API_BASE"] = env_base
    if env_key is not None:
        env["LLAMAFILE_API_KEY"] = env_key

    patch_base = patch.object(
        LlamafileChatConfig,
        "_resolve_api_base",
        wraps=LlamafileChatConfig._resolve_api_base,
    )
    patch_key = patch.object(
        LlamafileChatConfig,
        "_resolve_api_key",
        wraps=LlamafileChatConfig._resolve_api_key,
    )

    with patch.dict("os.environ", env, clear=True), patch_base as mock_base, patch_key as mock_key:
        result_base, result_key = config._get_openai_compatible_provider_info(
            api_base, api_key
        )

        assert result_base == expected_base
        assert result_key == expected_key

        mock_base.assert_called_once_with(api_base)
        mock_key.assert_called_once_with(api_key)


def test_completion_with_custom_llamafile_model():
    with patch(
        "litellm.main.openai_chat_completions.completion"
    ) as mock_llamafile_completion_func:
        mock_llamafile_completion_func.return_value = (
            {}
        )  # Return an empty dictionary for the mocked response

        provider = "llamafile"
        model_name = "my-custom-test-model"
        model = f"{provider}/{model_name}"
        messages = [{"role": "user", "content": "Hey, how's it going?"}]

        _ = litellm.completion(
            model=model,
            messages=messages,
            max_retries=2,
            max_tokens=100,
        )

        mock_llamafile_completion_func.assert_called_once()
        _, call_kwargs = mock_llamafile_completion_func.call_args
        assert call_kwargs.get("custom_llm_provider") == provider
        assert call_kwargs.get("model") == model_name
        assert call_kwargs.get("messages") == messages
        assert call_kwargs.get("api_base") == "http://127.0.0.1:8080/v1"
        assert call_kwargs.get("api_key") == "fake-api-key"
        optional_params = call_kwargs.get("optional_params")
        assert optional_params
        assert optional_params.get("max_retries") == 2
        assert optional_params.get("max_tokens") == 100
