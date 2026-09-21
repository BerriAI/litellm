"""
Unit tests for Nebius AI Studio configuration.

These tests validate the NebiusConfig class which extends OpenAIGPTConfig.
Nebius AI Studio is an OpenAI-compatible provider with minor customizations.
"""



from typing import Final

import pytest

import litellm
from litellm import completion
from litellm.llms.nebius.chat.transformation import NebiusConfig
from litellm.types.utils import ModelResponse, Usage


class TestNebiusConfig:
    """Test class for Nebius AI Studio functionality"""

    def test_default_api_base(self):
        """Test that default API base is used when none is provided"""
        config = NebiusConfig()
        headers = {}
        api_key = "fake-nebius-key"

        # Call validate_environment without specifying api_base
        result = config.validate_environment(
            headers=headers,
            model="nebius/Qwen/Qwen3-4B",
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
    def test_nebius_completion_mock(self, respx_mock):
        """
        Mock test for Nebius AI Studio completion using the model format from docs.
        This test mocks the actual HTTP request to test the integration properly.
        """

        litellm.disable_aiohttp_transport = (
            True  # since this uses respx, we need to set use_aiohttp_transport to False
        )

        # Set up environment variables for the test
        api_key = "fake-nebius-key"
        api_base = "https://api.studio.nebius.ai/v1"
        model = "nebius/Qwen/Qwen3-4B"
        model_name = "Qwen3-4B"  # The actual model name without provider prefix

        # Mock the HTTP request to the Nebius AI Studio API
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
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert hasattr(response.choices[0].message, "content")
        assert response.choices[0].message.content is not None

        # Check for specific content in the response
        assert "```python" in response.choices[0].message.content
        assert "Hey from LiteLLM" in response.choices[0].message.content


_NEBIUS_KIMI_K3: Final = "nebius/moonshotai/Kimi-K3"


def _nebius_cost_map_keys() -> set[str]:
    return {
        key
        for key, value in litellm.model_cost.items()
        if isinstance(value, dict) and value.get("litellm_provider") == "nebius"
    }


def _kimi_k3_row() -> dict:
    row: Final = litellm.model_cost[_NEBIUS_KIMI_K3]
    assert isinstance(row, dict)
    return row


class TestNebiusCostTracking:
    def test_cost_map_nebius_rows_are_registered_on_the_provider(self):
        registered: Final = set(litellm.models_by_provider.get("nebius") or ())
        assert _nebius_cost_map_keys() <= registered
        assert _NEBIUS_KIMI_K3 in registered

    def test_short_response_model_id_resolves_to_the_cost_map_row(self):
        row: Final = _kimi_k3_row()
        leaf: Final = _NEBIUS_KIMI_K3.rsplit("/", 1)[-1]
        info: Final = litellm.get_model_info(model=leaf, custom_llm_provider="nebius")
        assert info["input_cost_per_token"] == row["input_cost_per_token"]
        assert info["output_cost_per_token"] == row["output_cost_per_token"]

    def test_short_response_model_id_resolution_is_case_insensitive(self):
        row: Final = _kimi_k3_row()
        leaf: Final = _NEBIUS_KIMI_K3.rsplit("/", 1)[-1]
        swapped: Final = leaf.swapcase()
        assert swapped != leaf
        info: Final = litellm.get_model_info(model=swapped, custom_llm_provider="nebius")
        assert info["input_cost_per_token"] == row["input_cost_per_token"]
        assert info["output_cost_per_token"] == row["output_cost_per_token"]
        assert NebiusConfig().get_model_cost_key(_NEBIUS_KIMI_K3.lower()) == _NEBIUS_KIMI_K3

    def test_ambiguous_leaf_suffix_logs_and_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        import logging

        leaf: Final = "shared-leaf"
        monkeypatch.setattr(
            litellm,
            "nebius_models",
            {f"nebius/org-a/{leaf}", f"nebius/org-b/{leaf}"},
        )
        monkeypatch.setattr(litellm, "nebius_embedding_models", set())
        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            matched: Final = NebiusConfig().get_model_cost_key(leaf)
        assert matched is None
        assert "2 keys ending in /shared-leaf" in caplog.text
        assert "nebius/org-a/shared-leaf" in caplog.text
        assert "nebius/org-b/shared-leaf" in caplog.text

    def test_unmapped_leaf_returns_none_without_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        import logging

        monkeypatch.setattr(litellm, "nebius_models", set())
        monkeypatch.setattr(litellm, "nebius_embedding_models", set())
        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            matched: Final = NebiusConfig().get_model_cost_key("unmapped-leaf")
        assert matched is None
        assert "not using a suffix match" not in caplog.text

    def test_completion_cost_matches_cost_map_row_not_provider_usage_cost(self):
        row: Final = _kimi_k3_row()
        prompt_tokens: Final = 1_000_000
        completion_tokens: Final = 500_000
        map_cost: Final = (
            prompt_tokens * float(row["input_cost_per_token"])
            + completion_tokens * float(row["output_cost_per_token"])
        )
        response: Final = ModelResponse(
            model="Kimi-K3",
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost=map_cost / 10,
            ),
        )
        response._hidden_params["custom_llm_provider"] = "nebius"
        billed: Final = litellm.completion_cost(
            completion_response=response,
            model=_NEBIUS_KIMI_K3,
            custom_llm_provider="nebius",
        )
        assert billed == pytest.approx(map_cost)
        assert billed != pytest.approx(map_cost / 10)

    def test_stream_builder_prices_nebius_from_the_cost_map(self):
        from datetime import datetime

        from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

        row: Final = _kimi_k3_row()
        prompt_tokens: Final = 1_000_000
        completion_tokens: Final = 500_000
        map_cost: Final = (
            prompt_tokens * float(row["input_cost_per_token"])
            + completion_tokens * float(row["output_cost_per_token"])
        )
        usage_chunk: Final = ModelResponseStream(
            id="chatcmpl-nebius-cost",
            created=1724900000,
            model="Kimi-K3",
            object="chat.completion.chunk",
            choices=[StreamingChoices(finish_reason="stop", index=0, delta=Delta(content="", role="assistant"))],
        )
        usage_chunk.usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=map_cost / 10,
        )
        logging_obj: Final = LiteLLMLogging(
            model=_NEBIUS_KIMI_K3,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            call_type="completion",
            start_time=datetime.now(),
            litellm_call_id="test-nebius-cost",
            function_id="test-function-id",
        )
        logging_obj.update_environment_variables(
            model=_NEBIUS_KIMI_K3,
            user=None,
            optional_params={},
            litellm_params={"custom_llm_provider": "nebius"},
            custom_llm_provider="nebius",
        )
        response: Final = litellm.stream_chunk_builder(
            chunks=[usage_chunk],
            messages=[{"role": "user", "content": "hi"}],
            logging_obj=logging_obj,
        )
        assert response is not None
        assert response._hidden_params.get("response_cost") is None
        calculator_cost: Final = logging_obj._response_cost_calculator(result=response)
        client_cost: Final = getattr(response.usage, "cost", None)
        assert calculator_cost == pytest.approx(map_cost)
        assert client_cost == pytest.approx(calculator_cost)
        assert client_cost != pytest.approx(map_cost / 10)


class _FakeNebiusModelsResponse:
    status_code: Final = 200
    text: Final = ""

    def json(self) -> dict[str, list[dict[str, str]]]:
        return {"data": [{"id": "org/model-a"}, {"id": "org/model-b"}]}


class TestNebiusGetModels:
    def test_get_models_lists_from_live_v1_models(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}

        def mock_get(*, url: str, headers: dict[str, str]) -> _FakeNebiusModelsResponse:
            captured["url"] = url
            captured["headers"] = headers
            return _FakeNebiusModelsResponse()

        monkeypatch.setattr(litellm.module_level_client, "get", mock_get)
        models: Final = NebiusConfig().get_models(
            api_key="test-nebius-key",
            api_base="https://gateway.example/nebius/v1",
        )
        assert captured["url"] == "https://gateway.example/nebius/v1/models"
        assert captured["headers"] == {"Authorization": "Bearer test-nebius-key"}
        assert models == ["org/model-a", "org/model-b"]

    def test_get_models_uses_get_api_key_and_get_api_base(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NEBIUS_API_KEY", "env-nebius-key")
        monkeypatch.delenv("NEBIUS_API_BASE", raising=False)
        captured: dict[str, object] = {}

        def mock_get(*, url: str, headers: dict[str, str]) -> _FakeNebiusModelsResponse:
            captured["url"] = url
            captured["headers"] = headers
            return _FakeNebiusModelsResponse()

        monkeypatch.setattr(litellm.module_level_client, "get", mock_get)
        models: Final = NebiusConfig().get_models()
        assert captured["url"] == "https://api.studio.nebius.ai/v1/models"
        assert captured["headers"] == {"Authorization": "Bearer env-nebius-key"}
        assert models == ["org/model-a", "org/model-b"]

    def test_get_models_does_not_fall_back_to_openai_credentials(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

        def mock_get(**_kwargs: object) -> _FakeNebiusModelsResponse:
            raise AssertionError("must not call /v1/models without a Nebius key")

        monkeypatch.setattr(litellm.module_level_client, "get", mock_get)
        with pytest.raises(ValueError, match="NEBIUS_API_KEY"):
            NebiusConfig().get_models()

