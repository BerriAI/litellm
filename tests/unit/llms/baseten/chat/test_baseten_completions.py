import os
from typing import Final
from unittest.mock import patch

import httpx
import pytest
import respx

import litellm
from litellm.llms.baseten.chat import BasetenConfig
from litellm.utils import get_valid_models


class TestBasetenRouting:
    """Test Baseten routing logic"""

    def test_routing_logic(self):
        """Test routing between Model API and dedicated deployments"""
        config: Final = BasetenConfig()

        # Dedicated deployment (8-character alphanumeric)
        assert (
            config.get_api_base_for_model("abcd1234")
            == "https://model-abcd1234.api.baseten.co/environments/production/sync/v1"
        )

        # Model API (non-8-character)
        assert config.get_api_base_for_model("openai/gpt-oss-120b") == "https://inference.baseten.co/v1"


class TestBasetenModelAPI:
    """Test Baseten Model API inference"""

    @patch.dict(os.environ, {"BASETEN_API_KEY": "test-key"})
    def test_model_api_inference(self):
        """Test Model API inference with basic parameters"""
        config: Final = BasetenConfig()

        # Test parameter mapping
        non_default_params: Final = {"max_tokens": 100, "temperature": 0.7, "top_p": 0.9}

        result: Final = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="openai/gpt-oss-120b",
            drop_params=False,
        )

        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9

        # Test provider info
        api_base, api_key = config._get_openai_compatible_provider_info(None, "test-key")
        assert api_base == "https://inference.baseten.co/v1"
        assert api_key == "test-key"


class TestBasetenModelDiscovery:
    """Test Baseten model listing for `baseten/*` wildcard expansion and /v1/models"""

    def test_models_by_provider_comes_from_registry(self):
        """Static listing uses the model cost map, not the retired 2023 deployment IDs."""
        baseten_models: Final = litellm.models_by_provider["baseten"]
        servable_registry_models: Final = {
            key
            for key, value in litellm.model_cost.items()
            if value.get("litellm_provider") == "baseten" and not litellm._is_past_deprecation_date(value)
        }

        assert servable_registry_models
        assert servable_registry_models <= set(baseten_models)
        for retired_id in ("qvv0xeq", "q841o8w", "31dxrj3"):
            assert retired_id not in baseten_models

    def test_models_by_provider_skips_deprecated_models(self):
        """Retired Model APIs stay in the cost map for spend tracking but are not listed."""
        assert litellm.model_cost["baseten/zai-org/GLM-5"]["deprecation_date"]
        assert "baseten/zai-org/GLM-5" not in litellm.models_by_provider["baseten"]

    def test_cost_map_reload_drops_a_newly_retired_model(self):
        """A refreshed cost map that marks a listed model retired removes it without a restart."""
        model: Final = "baseten/acme/reload-probe"
        try:
            litellm.add_known_models({model: {"litellm_provider": "baseten"}})
            assert model in litellm.models_by_provider["baseten"]

            litellm.add_known_models({model: {"litellm_provider": "baseten", "deprecation_date": "2000-01-01"}})

            assert model not in litellm.models_by_provider["baseten"]
        finally:
            litellm.baseten_models.discard(model)
            litellm.add_known_models({})

    @pytest.mark.parametrize(
        "value, expected",
        [
            ({}, False),
            ({"deprecation_date": "2000-01-01"}, True),
            ({"deprecation_date": "2999-01-01"}, False),
            ({"deprecation_date": "not-a-date"}, False),
        ],
    )
    def test_is_past_deprecation_date(self, value, expected):
        assert litellm._is_past_deprecation_date(value) is expected

    def test_get_models_calls_model_api(self, respx_mock: respx.MockRouter) -> None:
        route: Final = respx_mock.get("https://inference.baseten.co/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "zai-org/GLM-5", "object": "model"},
                        {"id": "moonshotai/Kimi-K2.5", "object": "model"},
                    ],
                },
            )
        )

        models: Final = BasetenConfig().get_models(api_key="test-key")

        assert models == ["baseten/zai-org/GLM-5", "baseten/moonshotai/Kimi-K2.5"]
        assert route.calls.last.request.headers["Authorization"] == "Bearer test-key"

    def test_get_models_respects_custom_api_base(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get("https://custom.baseten.example/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "zai-org/GLM-5"}]})
        )

        models: Final = BasetenConfig().get_models(api_key="test-key", api_base="https://custom.baseten.example/v1/")

        assert models == ["baseten/zai-org/GLM-5"]

    @patch.dict(os.environ, {}, clear=True)
    def test_get_models_requires_api_key(self):
        with pytest.raises(ValueError, match="BASETEN_API_KEY"):
            BasetenConfig().get_models()

    def test_get_models_raises_on_error_status(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get("https://inference.baseten.co/v1/models").mock(
            return_value=httpx.Response(403, text="forbidden")
        )

        with pytest.raises(ValueError, match="403"):
            BasetenConfig().get_models(api_key="test-key")

    @patch.dict(os.environ, {"BASETEN_API_KEY": "test-key"})
    def test_get_valid_models_check_provider_endpoint(self, respx_mock: respx.MockRouter) -> None:
        """check_provider_endpoint=True returns the live catalog instead of the static list."""
        respx_mock.get("https://inference.baseten.co/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "brand-new/model"}]})
        )

        models: Final = get_valid_models(check_provider_endpoint=True, custom_llm_provider="baseten")

        assert models == ["baseten/brand-new/model"]


class TestBasetenReasoningEffort:
    def test_reasoning_effort_passthrough_with_drop_params(self):
        config: Final = BasetenConfig()

        result: Final = config.map_openai_params(
            non_default_params={"reasoning_effort": "high", "max_tokens": 100},
            optional_params={},
            model="deepseek-ai/DeepSeek-V4-Flash-0731",
            drop_params=True,
        )

        assert result["reasoning_effort"] == "high"
        assert result["max_tokens"] == 100

    def test_reasoning_effort_none_passthrough(self):
        config: Final = BasetenConfig()

        result: Final = config.map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model="moonshotai/Kimi-K3",
            drop_params=True,
        )

        assert result["reasoning_effort"] == "none"


if __name__ == "__main__":
    pytest.main([__file__])
