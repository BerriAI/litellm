"""
Tests for the TopxAI provider configuration and integration.
"""

import json
from datetime import datetime, timezone
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm._internal_context import pinned_billing_time
from litellm.llms.custom_httpx.http_handler import HTTPHandler

# Provider endpoint: https://api.topxai.xyz/docs (verified 2026-09-19)
TOPXAI_BASE_URL: Final = "https://api.topxai.xyz/v1"


class TestTopxAIProviderConfig:
    def test_topxai_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "TOPXAI")
        assert LlmProviders.TOPXAI.value == "topxai"
        assert "topxai" in litellm.provider_list

    def test_topxai_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("topxai")

        topxai = JSONProviderRegistry.get("topxai")
        assert topxai is not None
        assert topxai.base_url == TOPXAI_BASE_URL
        assert topxai.api_key_env == "TOPXAI_API_KEY"
        assert topxai.api_base_env == "TOPXAI_API_BASE"
        assert "/v1/chat/completions" in topxai.supported_endpoints

    def test_topxai_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "topxai" in openai_compatible_providers

    def test_topxai_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="topxai/claude-sonnet-5-5",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "claude-sonnet-5-5"
        assert provider == "topxai"
        assert api_base == TOPXAI_BASE_URL

    def test_topxai_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="topxai/claude-sonnet-5-5",
            custom_llm_provider=None,
            api_base="https://relay.example.com/v1",
            api_key="sk-test",
        )

        assert provider == "topxai"
        assert api_base == "https://relay.example.com/v1"
        assert api_key == "sk-test"

    def test_topxai_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="gpt-6.2-sol",
            custom_llm_provider=None,
            api_base=TOPXAI_BASE_URL,
            api_key=None,
        )
        assert provider == "topxai"
        assert api_base == TOPXAI_BASE_URL

    def test_topxai_resolves_env_api_key(self, monkeypatch):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("topxai")
        assert provider is not None
        config = create_config_class(provider)()
        monkeypatch.setenv("TOPXAI_API_KEY", "sk-test")
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == TOPXAI_BASE_URL
        assert api_key == "sk-test"

    @pytest.mark.parametrize("text_completion", (False, True))
    def test_missing_key_fails_before_dispatch(self, monkeypatch: pytest.MonkeyPatch, text_completion: bool) -> None:
        monkeypatch.delenv("TOPXAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "unrelated-test-key")

        with respx.mock() as upstream:
            with pytest.raises(litellm.AuthenticationError, match="TOPXAI_API_KEY"):
                if text_completion:
                    litellm.text_completion(model="topxai/gpt-6.2-sol", prompt="Hello")
                else:
                    litellm.completion(model="topxai/gpt-6.2-sol", messages=[{"role": "user", "content": "Hello"}])
            assert not upstream.calls

    @pytest.mark.parametrize("explicit_key", (None, "explicit-topxai-test-key"))
    def test_completion_uses_topxai_credentials(
        self, monkeypatch: pytest.MonkeyPatch, explicit_key: str | None
    ) -> None:
        monkeypatch.setenv("TOPXAI_API_KEY", "configured-topxai-test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "unrelated-test-key")
        with respx.mock() as upstream:
            route: Final = upstream.post(f"{TOPXAI_BASE_URL}/chat/completions").respond(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-6.2-sol",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
                    ],
                },
            )
            result: Final = litellm.completion(
                model="topxai/gpt-6.2-sol", messages=[{"role": "user", "content": "Hello"}], api_key=explicit_key
            )
        assert result.choices[0].message.content == "OK"
        assert (
            route.calls.last.request.headers["Authorization"]
            == f"Bearer {explicit_key or 'configured-topxai-test-key'}"
        )

    def test_topxai_complete_url_appends_endpoint(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("topxai")
        assert provider is not None
        config = create_config_class(provider)()
        url = config.get_complete_url(
            api_base=TOPXAI_BASE_URL,
            api_key="sk-test",
            model="topxai/kimi-k3.1-pro-Abliterated",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == f"{TOPXAI_BASE_URL}/chat/completions"

    @pytest.mark.parametrize(
        "model",
        (
            "gpt-6.2-sol",
            "gpt-6.2-astra",
            "grok-4.7",
            "gpt-6.1-terra",
            "kimi-k3.1-pro-Abliterated",
            "deepseek-flash-Abliterated",
            "deepseek-flash-fast",
            "GLM-5.3-Flash-Abliterated",
        ),
    )
    @pytest.mark.parametrize("api_base", (None, "https://relay.example.com/v1/"))
    def test_topxai_responses_routes_and_transforms(
        self, monkeypatch: pytest.MonkeyPatch, api_base: str | None, model: str
    ) -> None:
        monkeypatch.setenv("TOPXAI_API_KEY", "sk-topxai-test")
        expected_base: Final = (api_base or TOPXAI_BASE_URL).rstrip("/")

        def respond(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            assert str(request.url) == f"{expected_base}/responses"
            assert request.headers["Authorization"] == "Bearer sk-topxai-test"
            payload: Final = json.loads(request.content)
            assert payload["model"] == model
            assert payload["input"] == "Reply with OK"
            return httpx.Response(
                200,
                json={
                    "id": "resp_topxai_test",
                    "object": "response",
                    "created_at": 1,
                    "model": model,
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_test",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "OK", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
                },
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            response: Final = litellm.responses(
                model=f"topxai/{model}",
                input="Reply with OK",
                api_base=api_base,
                client=HTTPHandler(client=client),
            )

        assert response.status == "completed"
        assert response.output[0].content[0].text == "OK"
        assert response.usage.input_tokens == 4
        assert response.usage.output_tokens == 1

    def test_topxai_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "sonnet",
                    "litellm_params": {
                        "model": "topxai/claude-sonnet-5-5",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "sonnet"


class TestTopxAIModelMetadata:
    # Catalog and capabilities: https://api.topxai.xyz/api/pricing (verified 2026-11-19)
    TOPXAI_MODELS: Final = (
        "topxai/claude-sonnet-5-5",
        "topxai/claude-opus-5-5",
        "topxai/claude-fable-5-2",
        "topxai/gpt-6.2-sol",
        "topxai/gpt-6.1-terra",
        "topxai/gpt-6.2-astra",
        "topxai/grok-4.7",
        "topxai/kimi-k3.1-pro-Abliterated",
        "topxai/GLM-5.3-Abliterated",
        "topxai/GLM-5.3-Flash-Abliterated",
        "topxai/deepseek-flash-Abliterated",
        "topxai/deepseek-flash-fast",
    )
    TEXT_ONLY_MODELS: Final = ("topxai/GLM-5.3-Abliterated",)
    # Only these routes publish a cache-write price; the rest bill cache writes as input.
    # The GLM, Kimi and DeepSeek private deployments have none (2026-11-08).
    CACHE_WRITE_MODELS: Final = (
        "topxai/claude-sonnet-5-5",
        "topxai/claude-opus-5-5",
        "topxai/claude-fable-5-2",
        "topxai/gpt-6.2-sol",
        "topxai/gpt-6.1-terra",
        "topxai/gpt-6.2-astra",
    )
    # The catalogue publishes no output limit for these
    NO_OUTPUT_LIMIT_MODELS: Final = ("topxai/grok-4.7",)
    # First premium token and output multiplier:
    # https://api.topxai.xyz/pricing/gpt-6.2-sol (2026-11-18)
    # https://api.topxai.xyz/pricing/gpt-6.1-terra (2026-11-18)
    # https://api.topxai.xyz/pricing/gpt-6.2-astra (2026-11-18)
    # https://api.topxai.xyz/pricing/grok-4.7 (2026-11-18)
    TIERED_MODELS: Final = (
        ("topxai/gpt-6.2-sol", 272_001, 1.5),
        ("topxai/gpt-6.1-terra", 272_001, 1.5),
        ("topxai/gpt-6.2-astra", 272_001, 1.5),
        ("topxai/grok-4.7", 200_000, 2.0),
    )
    # Fixed prices at all hours (2026-11-08):
    # https://api.topxai.xyz/pricing/deepseek-flash-abliterated
    # https://api.topxai.xyz/pricing/deepseek-flash-fast
    DEEPSEEK_MODELS: Final = ("topxai/deepseek-flash-Abliterated", "topxai/deepseek-flash-fast")
    DEEPSEEK_MOMENTS: Final = (
        pytest.param(datetime(2026, 10, 26, 1, 0, tzinfo=timezone.utc), id="monday-01:00"),
        pytest.param(datetime(2026, 10, 26, 9, 59, tzinfo=timezone.utc), id="monday-09:59"),
        pytest.param(datetime(2026, 10, 26, 10, 0, tzinfo=timezone.utc), id="monday-10:00"),
        pytest.param(datetime(2026, 10, 25, 8, 0, tzinfo=timezone.utc), id="sunday-08:00"),
    )
    # Auto-route prices in USD per million tokens: input, output, cache read, cache write
    # (None where the route publishes no cache-write price). Claude, GPT and Grok are sold
    # on one route per lab at 80% of the lab's list price, long-context tiers included;
    # the GLM, Kimi and DeepSeek private deployments have fixed prices.
    # https://api.topxai.xyz/api/pricing, route_prices[auto_route] (verified 2026-11-19)
    AUTO_ROUTE_PRICES: Final = (
        ("topxai/claude-sonnet-5-5", 1.6, 8, 0.08, 2),
        ("topxai/claude-opus-5-5", 3.2, 16, 0.16, 4),
        ("topxai/claude-fable-5-2", 6.4, 32, 0.16, 8),
        ("topxai/gpt-6.2-sol", 1.6, 8, 0.08, 2),
        ("topxai/gpt-6.1-terra", 0.8, 4, 0.04, 1),
        ("topxai/gpt-6.2-astra", 6.4, 32, 0.32, 8),
        ("topxai/grok-4.7", 1.6, 4.8, 0.4, None),
        ("topxai/kimi-k3.1-pro-Abliterated", 3.25, 6.5, 0.325, None),
        ("topxai/GLM-5.3-Abliterated", 3, 6, 0.45, None),
        ("topxai/GLM-5.3-Flash-Abliterated", 0.36, 1.5, 0.075, None),
        ("topxai/deepseek-flash-Abliterated", 0.36, 1.35, 0.075, None),
        ("topxai/deepseek-flash-fast", 0.02, 0.16, 0.004, None),
    )

    @staticmethod
    def _load(path_parts):
        import json
        from pathlib import Path

        json_path = Path(__file__).parents[4].joinpath(*path_parts)
        with open(json_path) as f:
            return json.load(f)

    def test_topxai_models_registered_with_correct_metadata(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        for model in self.TOPXAI_MODELS:
            info = model_cost.get(model)
            assert info is not None, f"{model} missing from model_prices_and_context_window.json"
            assert info["litellm_provider"] == "topxai"
            assert info["mode"] == "chat"
            native_responses = litellm.ProviderConfigManager.get_provider_responses_api_config(
                litellm.LlmProviders.TOPXAI, model
            )
            assert (native_responses is not None) is ("/v1/responses" in info["supported_endpoints"])
            assert info["input_cost_per_token"] > 0
            assert info["output_cost_per_token"] > 0
            assert info["supports_function_calling"] is True
            assert info["supports_reasoning"] is True
            assert info["supports_vision"] is (model not in self.TEXT_ONLY_MODELS)

            assert info["supports_prompt_caching"] is True
            assert 0 < info["cache_read_input_token_cost"] < info["input_cost_per_token"]
            assert ("cache_creation_input_token_cost" in info) is (model in self.CACHE_WRITE_MODELS)
            assert (
                info.get("cache_creation_input_token_cost", info["input_cost_per_token"])
                >= info["input_cost_per_token"]
            )

            assert ("max_output_tokens" in info) is (model not in self.NO_OUTPUT_LIMIT_MODELS)
            assert info["max_input_tokens"] >= 500_000
            assert info["source"].startswith("https://api.topxai.xyz/pricing/")

    def test_topxai_auto_route_prices_cover_every_model(self):
        assert tuple(row[0] for row in self.AUTO_ROUTE_PRICES) == self.TOPXAI_MODELS

    @pytest.mark.parametrize("model,input_price,output_price,cache_read_price,cache_write_price", AUTO_ROUTE_PRICES)
    def test_topxai_auto_route_prices(
        self,
        model: str,
        input_price: float,
        output_price: float,
        cache_read_price: float,
        cache_write_price: "float | None",
    ) -> None:
        info: Final = self._load(("model_prices_and_context_window.json",))[model]
        per_million: Final = 1_000_000
        assert info["input_cost_per_token"] * per_million == pytest.approx(input_price)
        assert info["output_cost_per_token"] * per_million == pytest.approx(output_price)
        assert info["cache_read_input_token_cost"] * per_million == pytest.approx(cache_read_price)
        if cache_write_price is None:
            assert "cache_creation_input_token_cost" not in info
        else:
            assert info["cache_creation_input_token_cost"] * per_million == pytest.approx(cache_write_price)

    @pytest.mark.parametrize("model,first_premium_token,output_multiplier", TIERED_MODELS)
    @pytest.mark.parametrize("offset", (-1, 0, 1))
    @pytest.mark.parametrize("cached_tokens,cache_write_tokens", ((0, 0), (1000, 0), (0, 500), (1000, 500)))
    def test_topxai_cost_at_context_boundary(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model: str,
        first_premium_token: int,
        output_multiplier: float,
        offset: int,
        cached_tokens: int,
        cache_write_tokens: int,
    ) -> None:
        info: Final = self._load(("model_prices_and_context_window.json",))[model]
        monkeypatch.setitem(litellm.model_cost, model, info)
        prompt_tokens: Final = first_premium_token + offset
        completion_tokens: Final = 17
        # Whole-request input/cache rates double at these boundaries; sources above.
        # Without a published cache-write price a cache write bills as input.
        input_multiplier: Final = 2 if offset >= 0 else 1
        expected_input: Final = input_multiplier * (
            (prompt_tokens - cached_tokens - cache_write_tokens) * info["input_cost_per_token"]
            + cached_tokens * info["cache_read_input_token_cost"]
            + cache_write_tokens * info.get("cache_creation_input_token_cost", info["input_cost_per_token"])
        )
        expected_output: Final = (
            completion_tokens * info["output_cost_per_token"] * (output_multiplier if offset >= 0 else 1)
        )
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_input_tokens=cached_tokens,
            cache_creation_input_tokens=cache_write_tokens,
        )
        assert prompt_cost == pytest.approx(expected_input)
        assert completion_cost == pytest.approx(expected_output)

    @pytest.mark.parametrize("model", DEEPSEEK_MODELS)
    @pytest.mark.parametrize("moment", DEEPSEEK_MOMENTS)
    def test_topxai_deepseek_uses_fixed_prices_at_all_hours(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model: str,
        moment: datetime,
    ) -> None:
        info: Final = self._load(("model_prices_and_context_window.json",))[model]
        monkeypatch.setitem(litellm.model_cost, model, info)
        with pinned_billing_time(moment):
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=model,
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
                cache_read_input_tokens=400_000,
            )
        assert prompt_cost == pytest.approx(
            600_000 * info["input_cost_per_token"] + 400_000 * info["cache_read_input_token_cost"]
        )
        assert completion_cost == pytest.approx(1_000_000 * info["output_cost_per_token"])

    def test_topxai_models_synced_to_backup(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        backup = self._load(("litellm", "model_prices_and_context_window_backup.json"))
        for model in self.TOPXAI_MODELS:
            assert model in backup, f"{model} missing from backup json"
            assert backup[model] == model_cost[model], f"{model} differs between root and backup json"


class TestTopxAIDashboardRegistration:
    @staticmethod
    def _provider_create_fields():
        import json
        from pathlib import Path

        import litellm

        path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
        with open(path) as f:
            return json.load(f)

    def test_topxai_is_selectable_in_the_add_model_form(self):
        entries = [e for e in self._provider_create_fields() if e["litellm_provider"] == "topxai"]
        assert len(entries) == 1, "topxai must appear exactly once in provider_create_fields.json"

        entry = entries[0]
        assert entry["provider"] == "TOPXAI"
        assert entry["provider_display_name"] == "TopxAI"
        assert entry["default_model_placeholder"].startswith("topxai/")

        fields = {f["key"]: f for f in entry["credential_fields"]}
        assert fields["api_key"]["required"] is True
        assert fields["api_key"]["field_type"] == "password"
        assert fields["api_base"]["required"] is False
