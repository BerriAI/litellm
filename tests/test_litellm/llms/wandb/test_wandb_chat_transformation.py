"""
Unit tests for WandB Inference configuration.

These tests validate the WandbInferenceConfig class which extends OpenAIGPTConfig.
Nebius AI Studio is an OpenAI-compatible provider with minor customizations.
"""

import json
from typing import Final

import pytest
import respx

import litellm
from litellm import completion
from litellm.llms.wandb.chat.transformation import WandbConfig


WANDB_REASONING_MODELS: Final = (
    "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-ai/DeepSeek-V4-Pro-0813",
    "deepseek-ai/DeepSeek-V3.1",
    "google/gemma-4-31B-it",
    "ibm-granite/granite-4.2-8b",
    "MiniMaxAI/MiniMax-M3",
    "moonshotai/Kimi-K2.7-Code",
    "moonshotai/Kimi-K2.6",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3.6-27B",
    "Qwen/Qwen3.5-35B-A3B",
    "zai-org/GLM-5.2",
    "moonshotai/Kimi-K2.5",
    "MiniMaxAI/MiniMax-M2.5",
    "zai-org/GLM-4.5",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "deepseek-ai/DeepSeek-R1-0528",
)


@pytest.fixture
def wandb_test_config(local_model_cost_map, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "telemetry", False)
    monkeypatch.setattr(litellm, "drop_params", False)


@pytest.fixture
def wandb_request_mock(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post("https://api.inference.wandb.ai/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        status_code=200,
    )


class TestWandbConfig:
    """Test class for WandB Inference functionality"""

    @pytest.mark.parametrize("model", WANDB_REASONING_MODELS)
    def test_map_openai_params_preserves_reasoning_effort(self, wandb_test_config, model: str):
        assert litellm.model_cost[f"wandb/{model}"].get("supports_reasoning") is True
        supported_params = litellm.get_supported_openai_params(model=f"wandb/{model}")
        assert supported_params is not None
        assert "reasoning_effort" in supported_params

        result = WandbConfig().map_openai_params(
            non_default_params={"reasoning_effort": "medium", "max_completion_tokens": 64},
            optional_params={},
            model=model,
            drop_params=True,
        )

        assert result == {"reasoning_effort": "medium", "max_tokens": 64}

    def test_default_api_base(self):
        """Test that default API base is used when none is provided"""
        config = WandbConfig()
        headers = {}
        api_key = "fake-wandb-key"

        # Call validate_environment without specifying api_base
        result = config.validate_environment(
            headers=headers,
            model="wandb/openai/gpt-oss-20b",
            messages=[{"role": "user", "content": "Hey"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,  # Not providing api_base
        )

        # Verify headers are still set correctly
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

        # We can't directly test the api_base value here since validate_environment
        # only returns the headers, but we can verify it doesn't raise an exception
        # which would happen if api_base handling was incorrect

    @pytest.mark.respx()
    def test_wandb_completion_mock(self, respx_mock):
        """
        Mock test for WandB Inference completion using the model format from docs.
        This test mocks the actual HTTP request to test the integration properly.
        """

        litellm.disable_aiohttp_transport = (
            True  # since this uses respx, we need to set use_aiohttp_transport to False
        )

        # Set up environment variables for the test
        api_key = "fake-wandb-key"
        api_base = "https://api.inference.wandb.ai/v1"
        model = "wandb/openai/gpt-oss-20b"
        model_name = "gpt-oss-20b"  # The actual model name without provider prefix

        # Mock the HTTP request to the WandB Inference API
        respx_mock.post(f"{api_base}/chat/completions").respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```python\nprint("Hey from LiteLLM!")\n```\n\nThis simple Python code prints a greeting message from LiteLLM.',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            },
            status_code=200,
        )

        # Make the actual API call through LiteLLM
        response = completion(
            model=model,
            messages=[
                {"role": "user", "content": "write code for saying hey from LiteLLM"}
            ],
            api_key=api_key,
            api_base=api_base,
        )

        # Verify response structure
        assert response is not None

        # If response is a streaming wrapper, extract the first chunk for assertions
        # This handles both streaming and non-streaming responses
        # For streaming, response is typically an iterator yielding (event, data) tuples
        if hasattr(response, "__iter__") and not hasattr(response, "choices"):
            # Streaming response: get the first chunk
            first_chunk = next(iter(response))
            # first_chunk is likely a tuple: (event, data)
            # Try to extract the data part
            if isinstance(first_chunk, tuple) and len(first_chunk) == 2:
                data = first_chunk[1]
            else:
                data = first_chunk

            # The data object should have .choices[0] with .delta or .message
            choices = getattr(data, "choices", None)
            assert choices is not None
            assert len(choices) > 0
            choice = choices[0]
            # For streaming, content may be in .delta or .message
            content = None
            if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                content = choice.delta.content
            elif hasattr(choice, "message") and hasattr(choice.message, "content"):
                content = choice.message.content
            assert content is not None
            assert "```python" in content
            assert "Hey from LiteLLM" in content
        else:
            # Non-streaming response
            choices = getattr(response, "choices", None)
            assert choices is not None
            assert len(choices) > 0
            choice = choices[0]
            message = getattr(choice, "message", None)
            assert message is not None
            content = getattr(message, "content", None)
            assert content is not None

            # Check for specific content in the response
            assert "```python" in content
            assert "Hey from LiteLLM" in content

    @pytest.mark.respx()
    @pytest.mark.parametrize(
        "model,effort",
        tuple((model, "medium") for model in WANDB_REASONING_MODELS)
        + (
            ("Qwen/Qwen3.8-27B", "low"),
            ("Qwen/Qwen3.8-27B", "xhigh"),
        ),
    )
    def test_wandb_completion_preserves_reasoning_effort_with_drop_params(
        self, wandb_test_config, wandb_request_mock: respx.Route, model: str, effort: str
    ):
        completion(
            model=f"wandb/{model}",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="fake-wandb-key",
            api_base="https://api.inference.wandb.ai/v1",
            reasoning_effort=effort,
            max_completion_tokens=64,
            drop_params=True,
        )

        assert wandb_request_mock.call_count == 1
        request_body = json.loads(wandb_request_mock.calls[0].request.content)
        assert request_body["model"] == model
        assert request_body["reasoning_effort"] == effort
        assert request_body["max_tokens"] == 64
        assert "max_completion_tokens" not in request_body

    @pytest.mark.respx(assert_all_called=False)
    @pytest.mark.parametrize("drop_params", [True, False])
    @pytest.mark.parametrize(
        "model,explicit_false",
        [
            ("meta-llama/Llama-3.1-8B-Instruct", False),
            ("openai/gpt-oss-20b", True),
        ],
    )
    def test_wandb_completion_without_reasoning_support(
        self,
        wandb_test_config,
        wandb_request_mock: respx.Route,
        respx_mock: respx.MockRouter,
        monkeypatch: pytest.MonkeyPatch,
        model: str,
        explicit_false: bool,
        drop_params: bool,
    ):
        with monkeypatch.context() as context:
            if explicit_false:
                context.setitem(litellm.model_cost[f"wandb/{model}"], "supports_reasoning", False)

            kwargs = {
                "model": f"wandb/{model}",
                "messages": [{"role": "user", "content": "Hello"}],
                "api_key": "fake-wandb-key",
                "api_base": "https://api.inference.wandb.ai/v1",
                "reasoning_effort": "medium",
                "drop_params": drop_params,
            }
            if not drop_params:
                with pytest.raises(litellm.UnsupportedParamsError, match="reasoning_effort"):
                    completion(**kwargs)
                assert len(respx_mock.calls) == 0
                return

            completion(**kwargs)
            assert wandb_request_mock.call_count == 1
            request_body = json.loads(wandb_request_mock.calls[0].request.content)
            assert request_body["model"] == model
            assert "reasoning_effort" not in request_body

            supported_params = litellm.get_supported_openai_params(model=f"wandb/{model}")
            assert supported_params is not None
            assert "reasoning_effort" not in supported_params

    @pytest.mark.respx()
    def test_wandb_completion_keeps_reasoning_effort_for_an_unregistered_model(
        self, wandb_test_config, wandb_request_mock: respx.Route
    ):
        """A wandb id the registry has not named yet resolves through the
        wandb-reasoning-baseline fallback generalization, so its reasoning_effort reaches
        the provider instead of raising. W&B adds reasoning models faster than this
        registry names them, and an exact entry still wins wherever one exists."""
        model: Final = "zai-org/GLM-6-Turbo"
        assert f"wandb/{model}" not in litellm.model_cost

        completion(
            model=f"wandb/{model}",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="fake-wandb-key",
            api_base="https://api.inference.wandb.ai/v1",
            reasoning_effort="medium",
            drop_params=False,
        )

        assert wandb_request_mock.call_count == 1
        request_body = json.loads(wandb_request_mock.calls[0].request.content)
        assert request_body["model"] == model
        assert request_body["reasoning_effort"] == "medium"
