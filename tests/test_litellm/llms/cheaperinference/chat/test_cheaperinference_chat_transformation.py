"""
Unit tests for the Cheaper Inference provider configuration.

Cheaper Inference is an OpenAI-compatible discount gateway configured through
the JSON provider registry (`litellm/llms/openai_like/providers.json`), so
these tests cover registry loading, provider resolution, request
transformation and the declared Responses API support.

All tests are mock-based: no network calls and no API key are required.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

PROVIDER = "cheaperinference"
BASE_URL = "https://api.cheaperinference.com/v1"


class TestCheaperInferenceRegistry:
    def test_provider_is_registered(self):
        assert JSONProviderRegistry.exists(PROVIDER)

        provider = JSONProviderRegistry.get(PROVIDER)
        assert provider is not None
        assert provider.base_url == BASE_URL
        assert provider.api_key_env == "CHEAPERINFERENCE_API_KEY"
        assert provider.api_base_env == "CHEAPERINFERENCE_API_BASE"

    def test_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _api_key, api_base = get_llm_provider(
            model=f"{PROVIDER}/claude-sonnet-5",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "claude-sonnet-5"
        assert provider == PROVIDER
        assert api_base == BASE_URL

    def test_provider_config_manager_returns_config(self):
        from litellm import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(
            model="claude-sonnet-5", provider=LlmProviders.CHEAPERINFERENCE
        )

        assert config is not None
        assert config.custom_llm_provider == PROVIDER

    def test_responses_api_declared(self):
        assert JSONProviderRegistry.supports_responses_api(PROVIDER) is True


class TestCheaperInferenceTransformation:
    @pytest.fixture
    def config(self):
        provider = JSONProviderRegistry.get(PROVIDER)
        assert provider is not None
        return create_config_class(provider)()

    def test_default_api_base_and_auth_header(self, config):
        api_key = "ci_live_fake-key-for-tests"

        headers = config.validate_environment(
            headers={},
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "Hey"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,
        )

        assert headers["Authorization"] == f"Bearer {api_key}"
        assert headers["Content-Type"] == "application/json"

    def test_api_base_override(self, config):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == BASE_URL

        api_base, api_key = config._get_openai_compatible_provider_info(
            "https://proxy.internal/v1", "ci_live_fake-key-for-tests"
        )
        assert api_base == "https://proxy.internal/v1"
        assert api_key == "ci_live_fake-key-for-tests"

    @patch("litellm.utils.supports_function_calling", return_value=True)
    def test_supported_openai_params(self, _mock_supports_fc, config):
        supported = config.get_supported_openai_params(model="claude-sonnet-5")

        for param in ("temperature", "top_p", "max_tokens", "stream", "tools", "tool_choice"):
            assert param in supported


class TestCheaperInferenceCompletion:
    """End-to-end request/response handling with the HTTP layer mocked (no network)."""

    NON_STREAMING_RESPONSE = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1757462400,
        "model": "claude-sonnet-5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
    }

    @pytest.mark.respx()
    def test_completion_non_streaming(self, respx_mock):
        litellm.disable_aiohttp_transport = True

        route = respx_mock.post(f"{BASE_URL}/chat/completions").respond(
            json=self.NON_STREAMING_RESPONSE, status_code=200
        )

        response = litellm.completion(
            model=f"{PROVIDER}/claude-sonnet-5",
            messages=[{"role": "user", "content": "Hey"}],
            api_key="ci_live_fake-key-for-tests",
        )

        assert route.called
        request = route.calls[0].request
        assert request.headers["authorization"] == "Bearer ci_live_fake-key-for-tests"

        body = json.loads(request.content)
        assert body["model"] == "claude-sonnet-5"
        assert body["messages"] == [{"role": "user", "content": "Hey"}]

        assert response.choices[0].message.content == "Hello!"
        assert response.usage.total_tokens == 12

    @pytest.mark.respx()
    def test_completion_streaming(self, respx_mock):
        litellm.disable_aiohttp_transport = True

        chunks = [
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 1757462400,
                    "model": "claude-sonnet-5",
                    "choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}],
                }
            )
            + "\n\n",
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 1757462400,
                    "model": "claude-sonnet-5",
                    "choices": [{"index": 0, "delta": {"content": "lo!"}, "finish_reason": "stop"}],
                }
            )
            + "\n\n",
            "data: [DONE]\n\n",
        ]

        route = respx_mock.post(f"{BASE_URL}/chat/completions").respond(
            status_code=200,
            headers={"content-type": "text/plain"},
            content="".join(chunks),
        )

        response = litellm.completion(
            model=f"{PROVIDER}/claude-sonnet-5",
            messages=[{"role": "user", "content": "Hey"}],
            api_key="ci_live_fake-key-for-tests",
            stream=True,
        )

        received = list(response)

        assert route.called
        assert json.loads(route.calls[0].request.content)["stream"] is True
        assert "".join(chunk.choices[0].delta.content or "" for chunk in received) == "Hello!"


class TestCheaperInferenceCostTracking:
    @pytest.fixture(autouse=True)
    def local_cost_map(self, monkeypatch):
        """Read prices from the in-repo cost map instead of the published one."""
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
        litellm.get_model_info.cache_clear()
        yield
        litellm.get_model_info.cache_clear()

    @pytest.mark.parametrize(
        "model,input_cost,output_cost,cache_read_cost",
        [
            ("claude-sonnet-5", 1.400000 / 1_000_000, 7.000000 / 1_000_000, 0.140000 / 1_000_000),
            ("glm-5.3-flash", 0.105000 / 1_000_000, 0.350000 / 1_000_000, 0.012750 / 1_000_000),
        ],
    )
    def test_model_prices_registered(self, model, input_cost, output_cost, cache_read_cost):
        info = litellm.get_model_info(model=f"{PROVIDER}/{model}")

        assert info["litellm_provider"] == PROVIDER
        assert info["input_cost_per_token"] == pytest.approx(input_cost)
        assert info["output_cost_per_token"] == pytest.approx(output_cost)
        assert info["cache_read_input_token_cost"] == pytest.approx(cache_read_cost)

    @pytest.mark.respx()
    def test_cost_is_tracked_when_the_gateway_rewrites_the_model_name(self, respx_mock):
        """The gateway echoes the upstream model id, so costing must use the requested one.

        A real response to `cheaperinference/glm-5.3-flash` comes back with
        `"model": "z-ai/glm-5.3-flash"`. Cost has to follow the model the caller
        asked for, otherwise the lookup lands on an id that is not in the map.
        """
        litellm.disable_aiohttp_transport = True

        respx_mock.post(f"{BASE_URL}/chat/completions").respond(
            json={
                "id": "gen-test",
                "object": "chat.completion",
                "created": 1757462400,
                "model": "z-ai/glm-5.3-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 17, "completion_tokens": 3, "total_tokens": 20},
            },
            status_code=200,
        )

        response = litellm.completion(
            model=f"{PROVIDER}/glm-5.3-flash",
            messages=[{"role": "user", "content": "hi"}],
            api_key="ci_live_fake-key-for-tests",
        )

        expected = 17 * 0.105 / 1_000_000 + 3 * 0.350 / 1_000_000
        assert response._hidden_params["response_cost"] == pytest.approx(expected)

    def test_long_context_tier_registered(self):
        """Models the gateway prices in two bands carry the above-272k rates."""
        info = litellm.get_model_info(model=f"{PROVIDER}/gpt-5.6-luna")

        assert info["input_cost_per_token"] == pytest.approx(0.080000 / 1_000_000)
        assert info["input_cost_per_token_above_272k_tokens"] == pytest.approx(0.160000 / 1_000_000)
        assert info["output_cost_per_token_above_272k_tokens"] == pytest.approx(0.720000 / 1_000_000)


class TestCheaperInferenceResponsesStore:
    """The gateway's /v1/responses layer is stateless and rejects a request that does not send store=false."""

    @pytest.fixture
    def responses_config(self):
        from litellm.utils import ProviderConfigManager

        return ProviderConfigManager.get_provider_responses_api_config(
            provider=PROVIDER,
            model="claude-sonnet-5",
        )

    def test_registry_declares_force_store_false(self):
        provider = JSONProviderRegistry.get(PROVIDER)

        assert provider.special_handling.get("force_store_false") is True

    def test_store_is_forced_when_the_caller_omits_it(self, responses_config):
        from litellm.types.router import GenericLiteLLMParams

        transformed = responses_config.transform_responses_api_request(
            model="claude-sonnet-5",
            input="hello",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert transformed["store"] is False

    def test_store_true_from_the_caller_is_overridden(self, responses_config):
        from litellm.types.router import GenericLiteLLMParams

        params = {"store": True, "temperature": 0.2}
        transformed = responses_config.transform_responses_api_request(
            model="claude-sonnet-5",
            input="hello",
            response_api_optional_request_params=params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert transformed["store"] is False
        assert transformed["temperature"] == 0.2

    def test_responses_url_and_auth_header(self, responses_config):
        from litellm.types.router import GenericLiteLLMParams

        with patch.dict("os.environ", {"CHEAPERINFERENCE_API_KEY": "secret-from-env"}):
            headers = responses_config.validate_environment(
                headers={},
                model="claude-sonnet-5",
                litellm_params=GenericLiteLLMParams(),
            )

        assert headers["Authorization"] == "Bearer secret-from-env"
        assert (
            responses_config.get_complete_url(api_base=None, litellm_params={})
            == f"{BASE_URL}/responses"
        )
