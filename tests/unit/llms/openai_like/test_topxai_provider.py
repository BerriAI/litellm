"""
Tests for the TopxAI provider configuration and integration.
"""

import json
from datetime import datetime, timezone
from typing import Final

import httpx
import pytest

import litellm
from litellm._internal_context import pinned_billing_time
from litellm.llms.custom_httpx.http_handler import HTTPHandler

# Provider endpoint: https://ai.topxea.com/docs (verified 2026-09-19)
TOPXAI_BASE_URL: Final = "https://ai.topxea.com/v1"


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
        assert "/v1/responses" in topxai.supported_endpoints

    def test_topxai_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "topxai" in openai_compatible_providers

    def test_topxai_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="topxai/claude-sonnet-5",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "claude-sonnet-5"
        assert provider == "topxai"
        assert api_base == TOPXAI_BASE_URL

    def test_topxai_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="topxai/claude-sonnet-5",
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
            model="gpt-6-sol",
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

    def test_topxai_complete_url_appends_endpoint(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("topxai")
        assert provider is not None
        config = create_config_class(provider)()
        url = config.get_complete_url(
            api_base=TOPXAI_BASE_URL,
            api_key="sk-test",
            model="topxai/kimi-k3",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == f"{TOPXAI_BASE_URL}/chat/completions"

    @pytest.mark.parametrize("api_base", (None, "https://relay.example.com/v1/"))
    def test_topxai_responses_routes_and_transforms(
        self, monkeypatch: pytest.MonkeyPatch, api_base: str | None
    ) -> None:
        monkeypatch.setenv("TOPXAI_API_KEY", "sk-topxai-test")
        expected_base: Final = (api_base or TOPXAI_BASE_URL).rstrip("/")

        def respond(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            assert str(request.url) == f"{expected_base}/responses"
            assert request.headers["Authorization"] == "Bearer sk-topxai-test"
            payload: Final = json.loads(request.content)
            assert payload["model"] == "gpt-6-sol"
            assert payload["input"] == "Reply with OK"
            return httpx.Response(
                200,
                json={
                    "id": "resp_topxai_test",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-6-sol",
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
                model="topxai/gpt-6-sol",
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
                        "model": "topxai/claude-sonnet-5",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "sonnet"


class TestTopxAIModelMetadata:
    # Catalog and capabilities: https://ai.topxea.com/api/pricing (verified 2026-09-28)
    TOPXAI_MODELS: Final = (
        "topxai/claude-sonnet-5",
        "topxai/claude-opus-5-5",
        "topxai/claude-fable-5-1",
        "topxai/gpt-6-sol",
        "topxai/gpt-6-astra",
        "topxai/grok-4.7",
        "topxai/kimi-k3",
        "topxai/GLM-5.3-Abliterated",
        "topxai/deepseek-flash",
        "topxai/deepseek-v4-pro",
    )
    TEXT_ONLY_MODELS: Final = ("topxai/GLM-5.3-Abliterated", "topxai/deepseek-v4-pro")
    # Only these routes publish a cache-write price; the rest bill cache writes as input
    CACHE_WRITE_MODELS: Final = (
        "topxai/claude-sonnet-5",
        "topxai/claude-opus-5-5",
        "topxai/claude-fable-5-1",
        "topxai/gpt-6-sol",
        "topxai/gpt-6-astra",
        "topxai/kimi-k3",
    )
    # The catalogue publishes no output limit for these
    NO_OUTPUT_LIMIT_MODELS: Final = ("topxai/grok-4.7",)
    # First premium token and output multiplier, verified 2026-09-28:
    # https://ai.topxea.com/pricing/gpt-6-sol
    # https://ai.topxea.com/pricing/gpt-6-astra
    # https://ai.topxea.com/pricing/grok-4.7
    TIERED_MODELS: Final = (
        ("topxai/gpt-6-sol", 272_001, 1.5),
        ("topxai/gpt-6-astra", 272_001, 1.5),
        ("topxai/grok-4.7", 200_000, 2.0),
    )
    # USD for 1M prompt tokens (400K of them cache reads) and 1M completion tokens at the
    # deepseek-official peak lanes, verified 2026-09-28: flash $0.24 in, $0.0048 cached, $0.96 out;
    # v4-pro $1.056 in, $0.0352 cached, $3.168 out. Off-peak lanes are half.
    # https://ai.topxea.com/pricing/deepseek-flash
    # https://ai.topxea.com/pricing/deepseek-v4-pro
    DEEPSEEK_PEAK_COSTS: Final = (
        ("topxai/deepseek-flash", 1.10592),
        ("topxai/deepseek-v4-pro", 3.81568),
    )
    # Peak hours are 01:00-04:00 and 06:00-10:00 UTC, Monday to Friday
    DEEPSEEK_MOMENTS: Final = (
        pytest.param(datetime(2026, 9, 28, 1, 0, tzinfo=timezone.utc), 1.0, id="monday-01:00"),
        pytest.param(datetime(2026, 9, 28, 9, 59, tzinfo=timezone.utc), 1.0, id="monday-09:59"),
        pytest.param(datetime(2026, 9, 28, 0, 59, tzinfo=timezone.utc), 0.5, id="monday-00:59"),
        pytest.param(datetime(2026, 9, 28, 4, 0, tzinfo=timezone.utc), 0.5, id="monday-04:00"),
        pytest.param(datetime(2026, 9, 28, 10, 0, tzinfo=timezone.utc), 0.5, id="monday-10:00"),
        pytest.param(datetime(2026, 9, 27, 8, 0, tzinfo=timezone.utc), 0.5, id="sunday-08:00"),
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
            assert info["input_cost_per_token"] > 0
            assert info["output_cost_per_token"] > 0
            assert info["supports_function_calling"] is True
            assert info["supports_tool_choice"] is True
            assert info["supports_reasoning"] is True
            assert info["supports_response_schema"] is True
            assert info["supports_vision"] is (model not in self.TEXT_ONLY_MODELS)

            assert info["supports_prompt_caching"] is True
            assert 0 < info["cache_read_input_token_cost"] < info["input_cost_per_token"]
            assert ("cache_creation_input_token_cost" in info) is (model in self.CACHE_WRITE_MODELS)
            assert (
                info.get("cache_creation_input_token_cost", info["input_cost_per_token"])
                >= info["input_cost_per_token"]
            )

            assert ("max_output_tokens" in info) is (model not in self.NO_OUTPUT_LIMIT_MODELS)
            assert info.get("max_tokens") == info.get("max_output_tokens")
            assert info["max_input_tokens"] >= 500_000
            assert info["source"].startswith("https://ai.topxea.com/pricing/")

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

    @pytest.mark.parametrize("model,peak_cost", DEEPSEEK_PEAK_COSTS)
    @pytest.mark.parametrize("moment,rate_share", DEEPSEEK_MOMENTS)
    def test_topxai_deepseek_bills_peak_lanes_only_in_weekday_peak_hours(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model: str,
        peak_cost: float,
        moment: datetime,
        rate_share: float,
    ) -> None:
        monkeypatch.setitem(litellm.model_cost, model, self._load(("model_prices_and_context_window.json",))[model])
        with pinned_billing_time(moment):
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=model,
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
                cache_read_input_tokens=400_000,
            )
        assert prompt_cost + completion_cost == pytest.approx(peak_cost * rate_share)

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
