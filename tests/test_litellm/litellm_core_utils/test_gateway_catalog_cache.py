"""
Tests for gateway catalog metadata: fetchers, normalizers, TTL cache.

Fixtures mirror the real catalog shapes:
- OpenRouter: GET https://openrouter.ai/api/v1/models
- Vercel: GET https://ai-gateway.vercel.sh/v1/models
- Merge: GET https://api-gateway.merge.dev/v1/models (native catalog,
  paginated via has_more/next_cursor)
"""

import time
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.litellm_core_utils import gateway_catalog_cache
from litellm.litellm_core_utils.gateway_catalog_cache import (
    get_catalog,
    register_catalog_into_model_cost,
)
from litellm.llms.merge_ai_gateway.chat.transformation import MergeAIGatewayConfig
from litellm.llms.openrouter.chat.transformation import OpenrouterConfig
from litellm.llms.vercel_ai_gateway.chat.transformation import VercelAIGatewayConfig

OPENROUTER_ITEM = {
    "id": "anthropic/claude-sonnet-4",
    "context_length": 200000,
    "architecture": {
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
    },
    "pricing": {
        "prompt": "0.000003",
        "completion": "0.000015",
        "input_cache_read": "0.0000003",
        "input_cache_write": "0.00000375",
    },
    "top_provider": {"context_length": 200000, "max_completion_tokens": 64000},
    "supported_parameters": ["max_tokens", "reasoning", "tools"],
}

VERCEL_ITEM = {
    "id": "anthropic/claude-sonnet-4",
    "type": "language",
    "context_window": 200000,
    "max_tokens": 64000,
    "tags": ["reasoning", "tool-use"],
    "modalities": {"input": ["text", "image"], "output": ["text"]},
    "pricing": {"input": "0.000003", "output": "0.000015", "input_cache_read": "0.0000003"},
}

MERGE_ITEM = {
    "model": "anthropic/claude-opus-4-6",
    "provider": "anthropic",
    "display_name": "Claude Opus 4.6",
    "availability_status": "available",
    "vendors": {
        "anthropic": {
            "context_window": 200000,
            "max_output_tokens": 64000,
            "availability_status": "available",
            "capabilities": {
                "input": ["text", "image", "document"],
                "output": ["text", "tool_use"],
                "supports_tool_calling": True,
                "supports_tool_choice": True,
                "supports_structured_outputs": True,
                "streaming": True,
            },
            "pricing": {
                "currency": "USD",
                "input_per_million": 5.0,
                "output_per_million": 25.0,
                "flex": {"input_per_million": 2.5, "output_per_million": 12.5},
            },
            "service_tiers": ["standard", "flex"],
        },
        "bedrock": {
            "context_window": 200000,
            "max_output_tokens": 64000,
            "availability_status": "available",
            "capabilities": {
                "input": ["text"],
                "output": ["text"],
                "supports_tool_calling": False,
                "supports_tool_choice": False,
                "supports_structured_outputs": False,
                "streaming": True,
            },
            "pricing": {"currency": "USD", "input_per_million": 15.0, "output_per_million": 75.0},
        },
    },
}


def _mock_response(payload, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = payload
    mock.text = str(payload)
    return mock


@pytest.fixture(autouse=True)
def clear_catalog_cache():
    gateway_catalog_cache._catalog_cache.clear()
    yield
    gateway_catalog_cache._catalog_cache.clear()


class TestOpenRouterCatalog:
    def test_normalizes_pricing_and_capabilities(self):
        with patch(
            "litellm.module_level_client.get",
            return_value=_mock_response({"data": [OPENROUTER_ITEM, {"id": "~alias/model"}]}),
        ) as mock_get:
            entries = OpenrouterConfig().get_models_with_info(api_key="sk-or")

        assert mock_get.call_args.kwargs["url"] == "https://openrouter.ai/api/v1/models"
        assert len(entries) == 1  # ~alias skipped
        entry = entries[0]
        assert entry["key"] == "openrouter/anthropic/claude-sonnet-4"
        assert entry["litellm_provider"] == "openrouter"
        assert entry["mode"] == "chat"
        assert entry["max_input_tokens"] == 200000
        assert entry["max_output_tokens"] == 64000
        assert entry["input_cost_per_token"] == pytest.approx(3e-6)
        assert entry["output_cost_per_token"] == pytest.approx(15e-6)
        assert entry["cache_read_input_token_cost"] == pytest.approx(3e-7)
        assert entry["supports_vision"] is True
        assert entry["supports_reasoning"] is True


class TestVercelCatalog:
    def test_normalizes_and_filters_types(self):
        payload = {
            "data": [
                VERCEL_ITEM,
                {"id": "alibaba/qwen3-embedding-0.6b", "type": "embedding",
                 "context_window": 32768, "max_tokens": 32768,
                 "pricing": {"input": "0.00000001"}},
                {"id": "openai/dall-e-3", "type": "image"},
            ]
        }
        with patch("litellm.module_level_client.get", return_value=_mock_response(payload)) as mock_get:
            entries = VercelAIGatewayConfig().get_models_with_info()

        assert mock_get.call_args.kwargs["url"] == "https://ai-gateway.vercel.sh/v1/models"
        assert [e["key"] for e in entries] == [
            "vercel_ai_gateway/anthropic/claude-sonnet-4",
            "vercel_ai_gateway/alibaba/qwen3-embedding-0.6b",
        ]
        chat, embed = entries
        assert chat["mode"] == "chat"
        assert chat["max_output_tokens"] == 64000
        assert chat["input_cost_per_token"] == pytest.approx(3e-6)
        assert chat["supports_vision"] is True
        assert chat["supports_reasoning"] is True
        assert embed["mode"] == "embedding"
        assert embed["max_output_tokens"] is None


class TestMergeCatalog:
    def _pages(self, items, has_more=False, next_cursor=None):
        return _mock_response(
            {"object": "list", "data": items, "has_more": has_more, "next_cursor": next_cursor}
        )

    def test_selects_cheapest_available_vendor(self):
        with patch(
            "litellm.module_level_client.get",
            return_value=self._pages([MERGE_ITEM]),
        ) as mock_get:
            entries = MergeAIGatewayConfig().get_models_with_info(api_key="sk-merge")

        assert mock_get.call_args.kwargs["url"] == "https://api-gateway.merge.dev/v1/models"
        assert mock_get.call_args.kwargs["params"] == {"limit": 500}
        assert len(entries) == 1
        entry = entries[0]
        assert entry["key"] == "merge_ai_gateway/anthropic/claude-opus-4-6"
        # anthropic vendor (5/25 per million) beats bedrock (15/75)
        assert entry["input_cost_per_token"] == pytest.approx(5e-6)
        assert entry["output_cost_per_token"] == pytest.approx(25e-6)
        assert entry["input_cost_per_token_flex"] == pytest.approx(2.5e-6)
        assert entry["max_input_tokens"] == 200000
        assert entry["max_output_tokens"] == 64000
        assert entry["supports_vision"] is True
        assert entry["supports_pdf_input"] is True
        assert entry["supports_function_calling"] is True
        assert entry["supports_native_streaming"] is True

    def test_pagination_follows_next_cursor(self):
        first = self._pages([MERGE_ITEM], has_more=True, next_cursor="cur2")
        second = self._pages([{**MERGE_ITEM, "model": "openai/gpt-5"}])
        with patch("litellm.module_level_client.get", side_effect=[first, second]) as mock_get:
            entries = MergeAIGatewayConfig().get_models_with_info(api_key="sk-merge")

        assert mock_get.call_count == 2
        assert mock_get.call_args_list[1].kwargs["params"] == {"limit": 500, "cursor": "cur2"}
        assert {e["key"] for e in entries} == {
            "merge_ai_gateway/anthropic/claude-opus-4-6",
            "merge_ai_gateway/openai/gpt-5",
        }

    def test_skips_unavailable_and_embedding_only(self):
        unavailable = {**MERGE_ITEM, "model": "x/gone", "vendors": {
            "v1": {**MERGE_ITEM["vendors"]["anthropic"], "availability_status": "deprecated"}
        }}
        embed_only = {**MERGE_ITEM, "model": "x/embed", "vendors": {
            "v1": {**MERGE_ITEM["vendors"]["anthropic"], "capabilities": {
                "input": ["text"], "output": ["embedding"],
                "supports_tool_calling": False, "supports_tool_choice": False,
                "supports_structured_outputs": False, "streaming": False,
            }}
        }}
        with patch(
            "litellm.module_level_client.get",
            return_value=self._pages([unavailable, embed_only, MERGE_ITEM]),
        ):
            entries = MergeAIGatewayConfig().get_models_with_info(api_key="sk-merge")

        assert [e["key"] for e in entries] == ["merge_ai_gateway/anthropic/claude-opus-4-6"]

    def test_pricing_fallback_to_openrouter_cost_map(self):
        no_pricing = {
            **MERGE_ITEM,
            "model": "anthropic/claude-3-haiku",
            "vendors": {
                "v1": {
                    **MERGE_ITEM["vendors"]["anthropic"],
                    "pricing": {"currency": "USD"},
                }
            },
        }
        original = litellm.model_cost.get("openrouter/anthropic/claude-3-haiku")
        assert original is not None and original.get("input_cost_per_token") is not None

        with patch(
            "litellm.module_level_client.get",
            return_value=self._pages([no_pricing]),
        ):
            entries = MergeAIGatewayConfig().get_models_with_info(api_key="sk-merge")

        entry = entries[0]
        assert entry["input_cost_per_token"] == original["input_cost_per_token"]
        assert entry["output_cost_per_token"] == original["output_cost_per_token"]


class TestCatalogCache:
    def test_caches_within_ttl(self):
        with patch(
            "litellm.module_level_client.get",
            return_value=_mock_response({"data": [OPENROUTER_ITEM]}),
        ) as mock_get:
            first = get_catalog("openrouter", "sk-or", None)
            second = get_catalog("openrouter", "sk-or", None)

        assert mock_get.call_count == 1
        assert first is second
        assert "anthropic/claude-sonnet-4" in first

    def test_cache_key_separates_api_base(self):
        with patch(
            "litellm.module_level_client.get",
            return_value=_mock_response({"data": [OPENROUTER_ITEM]}),
        ) as mock_get:
            get_catalog("openrouter", "sk-or", None)
            get_catalog("openrouter", "sk-or", "https://other.example.com/v1")

        assert mock_get.call_count == 2

    def test_ttl_expiry_refetches(self):
        with patch(
            "litellm.module_level_client.get",
            return_value=_mock_response({"data": [OPENROUTER_ITEM]}),
        ) as mock_get:
            get_catalog("openrouter", "sk-or", None)
            key = next(iter(gateway_catalog_cache._catalog_cache))
            ts, value = gateway_catalog_cache._catalog_cache[key]
            gateway_catalog_cache._catalog_cache[key] = (ts - 400, value)
            get_catalog("openrouter", "sk-or", None)

        assert mock_get.call_count == 2

    def test_failed_fetch_returns_none(self):
        with patch(
            "litellm.module_level_client.get",
            return_value=_mock_response({}, status_code=500),
        ):
            assert get_catalog("openrouter", "sk-or", None) is None

    def test_unknown_provider_returns_none(self):
        assert get_catalog("not_a_provider", None, None) is None


class TestRegisterCatalog:
    def test_registers_prefixed_keys(self):
        key = "merge/anthropic/claude-opus-4-6"
        try:
            register_catalog_into_model_cost(
                "merge",
                {"anthropic/claude-opus-4-6": {"input_cost_per_token": 5e-6, "mode": "chat"}},
            )
            assert litellm.model_cost[key]["input_cost_per_token"] == 5e-6
            assert litellm.model_cost[key]["key"] == key
        finally:
            litellm.model_cost.pop(key, None)
