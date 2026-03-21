from unittest.mock import patch

import litellm


def test_rapid_mlx_provider_routing():
    """Test that rapid_mlx/ prefix routes correctly as an OpenAI-compatible provider."""
    with patch(
        "litellm.main.openai_chat_completions.completion"
    ) as mock_completion:
        mock_completion.return_value = {}

        provider = "rapid_mlx"
        model_name = "default"
        model = f"{provider}/{model_name}"
        messages = [{"role": "user", "content": "Hello!"}]

        _ = litellm.completion(
            model=model,
            messages=messages,
            max_tokens=100,
        )

        mock_completion.assert_called_once()
        _, call_kwargs = mock_completion.call_args
        assert call_kwargs.get("custom_llm_provider") == provider
        assert call_kwargs.get("model") == model_name
        assert call_kwargs.get("messages") == messages
        assert call_kwargs.get("api_base") == "http://localhost:8000/v1"
        assert call_kwargs.get("api_key") == "not-needed"


def test_rapid_mlx_custom_api_base():
    """Test that RAPID_MLX_API_BASE environment variable is respected."""
    with patch(
        "litellm.main.openai_chat_completions.completion"
    ) as mock_completion, patch.dict(
        "os.environ",
        {"RAPID_MLX_API_BASE": "http://192.168.1.100:8000/v1"},
    ):
        mock_completion.return_value = {}

        _ = litellm.completion(
            model="rapid_mlx/qwen3.5-9b",
            messages=[{"role": "user", "content": "test"}],
        )

        mock_completion.assert_called_once()
        _, call_kwargs = mock_completion.call_args
        assert call_kwargs.get("api_base") == "http://192.168.1.100:8000/v1"


def test_rapid_mlx_custom_api_key():
    """Test that RAPID_MLX_API_KEY environment variable is respected."""
    with patch(
        "litellm.main.openai_chat_completions.completion"
    ) as mock_completion, patch.dict(
        "os.environ",
        {"RAPID_MLX_API_KEY": "my-secret-key"},
    ):
        mock_completion.return_value = {}

        _ = litellm.completion(
            model="rapid_mlx/default",
            messages=[{"role": "user", "content": "test"}],
        )

        mock_completion.assert_called_once()
        _, call_kwargs = mock_completion.call_args
        assert call_kwargs.get("api_key") == "my-secret-key"
