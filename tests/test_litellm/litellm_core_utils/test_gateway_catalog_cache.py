"""
Tests for gateway catalog metadata: fetchers, normalizers, TTL cache.

Fixtures mirror the real catalog shapes:
- OpenRouter: GET https://openrouter.ai/api/v1/models
- Vercel: GET https://ai-gateway.vercel.sh/v1/models
- Merge: GET https://api-gateway.merge.dev/v1/models (native catalog,
  paginated via has_more/next_cursor)

HTTP is faked with respx at the transport boundary, so the assertions read the
requests the provider actually sends.
"""

from types import MappingProxyType

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils import gateway_catalog_cache
from litellm.litellm_core_utils.gateway_catalog_cache import (
    get_catalog,
    prefix_model_ids,
    register_catalog_into_model_cost,
)
from litellm.llms.merge_ai_gateway.chat.transformation import MergeAIGatewayConfig
from litellm.llms.openrouter.chat.transformation import OpenrouterConfig
from litellm.llms.vercel_ai_gateway.chat.transformation import VercelAIGatewayConfig

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
VERCEL_URL = "https://ai-gateway.vercel.sh/v1/models"
MERGE_URL = "https://api-gateway.merge.dev/v1/models"

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


def _catalog_page(items, has_more=False, next_cursor=None):
    return {"object": "list", "data": items, "has_more": has_more, "next_cursor": next_cursor}


@pytest.fixture(autouse=True)
def clear_catalog_cache():
    gateway_catalog_cache._CATALOG_CACHE.clear()
    yield
    gateway_catalog_cache._CATALOG_CACHE.clear()


class TestPrefixModelIds:
    def test_namespaces_bare_ids_and_keeps_the_upstream_org(self):
        assert prefix_model_ids("openrouter", ["anthropic/claude-sonnet-4", "openai/gpt-5"]) == [
            "openrouter/anthropic/claude-sonnet-4",
            "openrouter/openai/gpt-5",
        ]

    def test_does_not_double_prefix(self):
        assert prefix_model_ids("merge_ai_gateway", ["merge_ai_gateway/x/y"]) == ["merge_ai_gateway/x/y"]


class TestOpenRouterCatalog:
    def test_normalizes_pricing_and_capabilities(self, respx_mock):
        route = respx_mock.get(OPENROUTER_URL).mock(
            return_value=httpx.Response(200, json={"data": [OPENROUTER_ITEM, {"id": "~alias/model"}]})
        )

        catalog = OpenrouterConfig().get_models_with_info(api_key="sk-or")

        assert list(catalog) == ["anthropic/claude-sonnet-4"]  # ~alias skipped
        entry = catalog["anthropic/claude-sonnet-4"]
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
        assert route.calls[0].request.headers["authorization"] == "Bearer sk-or"

    def test_get_models_namespaces_ids(self, respx_mock):
        respx_mock.get(OPENROUTER_URL).mock(return_value=httpx.Response(200, json={"data": [OPENROUTER_ITEM]}))

        assert OpenrouterConfig().get_models() == ["openrouter/anthropic/claude-sonnet-4"]

    def test_survives_unexpected_field_types(self, respx_mock):
        """Catalog payloads drift; a wrong type must not raise."""
        payload = {
            "data": [
                {
                    "id": "vendor/model",
                    "context_length": "not-a-number",
                    "architecture": {"input_modalities": "text"},
                    "pricing": {"prompt": None, "completion": "free"},
                    "top_provider": "unexpected",
                    "supported_parameters": None,
                }
            ]
        }
        respx_mock.get(OPENROUTER_URL).mock(return_value=httpx.Response(200, json=payload))

        catalog = OpenrouterConfig().get_models_with_info()

        entry = catalog["vendor/model"]
        assert entry["max_input_tokens"] is None
        assert entry["input_cost_per_token"] is None
        assert entry["output_cost_per_token"] is None
        assert entry["max_output_tokens"] is None
        assert entry["supports_vision"] is False
        assert entry["supports_reasoning"] is False


class TestVercelCatalog:
    def test_normalizes_and_filters_types(self, respx_mock):
        payload = {
            "data": [
                VERCEL_ITEM,
                {
                    "id": "alibaba/qwen3-embedding-0.6b",
                    "type": "embedding",
                    "context_window": 32768,
                    "max_tokens": 32768,
                    "pricing": {"input": "0.00000001"},
                },
                {"id": "openai/dall-e-3", "type": "image"},
            ]
        }
        route = respx_mock.get(VERCEL_URL).mock(return_value=httpx.Response(200, json=payload))

        catalog = VercelAIGatewayConfig().get_models_with_info()

        assert route.called
        assert list(catalog) == ["anthropic/claude-sonnet-4", "alibaba/qwen3-embedding-0.6b"]
        chat = catalog["anthropic/claude-sonnet-4"]
        embed = catalog["alibaba/qwen3-embedding-0.6b"]
        assert chat["key"] == "vercel_ai_gateway/anthropic/claude-sonnet-4"
        assert chat["mode"] == "chat"
        assert chat["max_output_tokens"] == 64000
        assert chat["input_cost_per_token"] == pytest.approx(3e-6)
        assert chat["supports_vision"] is True
        assert chat["supports_reasoning"] is True
        assert embed["mode"] == "embedding"
        assert embed["max_output_tokens"] is None

    def test_get_models_namespaces_ids(self, respx_mock):
        respx_mock.get(VERCEL_URL).mock(return_value=httpx.Response(200, json={"data": [VERCEL_ITEM]}))

        assert VercelAIGatewayConfig().get_models() == ["vercel_ai_gateway/anthropic/claude-sonnet-4"]


class TestMergeCatalog:
    def test_selects_cheapest_available_vendor(self, respx_mock):
        route = respx_mock.get(MERGE_URL).mock(return_value=httpx.Response(200, json=_catalog_page([MERGE_ITEM])))

        catalog = MergeAIGatewayConfig().get_models_with_info(api_key="sk-merge")

        request = route.calls[0].request
        assert request.url.params["limit"] == "500"
        assert "cursor" not in request.url.params
        assert request.headers["authorization"] == "Bearer sk-merge"
        entry = catalog["anthropic/claude-opus-4-6"]
        assert entry["key"] == "merge_ai_gateway/anthropic/claude-opus-4-6"
        # anthropic vendor (5/25 per million) beats bedrock (15/75)
        assert entry["input_cost_per_token"] == pytest.approx(5e-6)
        assert entry["output_cost_per_token"] == pytest.approx(25e-6)
        assert entry["input_cost_per_token_flex"] == pytest.approx(2.5e-6)
        assert entry["output_cost_per_token_flex"] == pytest.approx(12.5e-6)
        assert entry["max_input_tokens"] == 200000
        assert entry["max_output_tokens"] == 64000
        assert entry["supports_vision"] is True
        assert entry["supports_pdf_input"] is True
        assert entry["supports_function_calling"] is True
        assert entry["supports_native_streaming"] is True

    def test_catalog_root_derived_from_openai_shaped_api_base(self):
        assert (
            MergeAIGatewayConfig.get_catalog_root("https://api-gateway.merge.dev/v1/openai")
            == "https://api-gateway.merge.dev/v1"
        )
        assert MergeAIGatewayConfig.get_catalog_root(None) == "https://api-gateway.merge.dev/v1"

    def test_pagination_follows_next_cursor(self, respx_mock):
        route = respx_mock.get(MERGE_URL).mock(
            side_effect=[
                httpx.Response(200, json=_catalog_page([MERGE_ITEM], has_more=True, next_cursor="cur2")),
                httpx.Response(200, json=_catalog_page([{**MERGE_ITEM, "model": "openai/gpt-5"}])),
            ]
        )

        catalog = MergeAIGatewayConfig().get_models_with_info(api_key="sk-merge")

        assert len(route.calls) == 2
        assert route.calls[1].request.url.params["cursor"] == "cur2"
        assert set(catalog) == {"anthropic/claude-opus-4-6", "openai/gpt-5"}

    def test_stops_when_next_cursor_missing(self, respx_mock):
        route = respx_mock.get(MERGE_URL).mock(
            return_value=httpx.Response(200, json=_catalog_page([MERGE_ITEM], has_more=True, next_cursor=None))
        )

        catalog = MergeAIGatewayConfig().get_models_with_info(api_key="sk-merge")

        assert list(catalog) == ["anthropic/claude-opus-4-6"]
        assert len(route.calls) == 1

    def test_skips_unavailable_and_embedding_only(self, respx_mock):
        unavailable = {
            **MERGE_ITEM,
            "model": "x/gone",
            "vendors": {"v1": {**MERGE_ITEM["vendors"]["anthropic"], "availability_status": "deprecated"}},
        }
        embed_only = {
            **MERGE_ITEM,
            "model": "x/embed",
            "vendors": {
                "v1": {
                    **MERGE_ITEM["vendors"]["anthropic"],
                    "capabilities": {
                        "input": ["text"],
                        "output": ["embedding"],
                        "supports_tool_calling": False,
                        "supports_tool_choice": False,
                        "supports_structured_outputs": False,
                        "streaming": False,
                    },
                }
            },
        }
        respx_mock.get(MERGE_URL).mock(
            return_value=httpx.Response(200, json=_catalog_page([unavailable, embed_only, MERGE_ITEM]))
        )

        catalog = MergeAIGatewayConfig().get_models_with_info(api_key="sk-merge")

        assert list(catalog) == ["anthropic/claude-opus-4-6"]

    def test_pricing_fallback_to_openrouter_cost_map(self, respx_mock):
        no_pricing = {
            **MERGE_ITEM,
            "model": "anthropic/claude-3-haiku",
            "vendors": {"v1": {**MERGE_ITEM["vendors"]["anthropic"], "pricing": {"currency": "USD"}}},
        }
        original = litellm.model_cost.get("openrouter/anthropic/claude-3-haiku")
        assert original is not None and original.get("input_cost_per_token") is not None
        respx_mock.get(MERGE_URL).mock(return_value=httpx.Response(200, json=_catalog_page([no_pricing])))

        catalog = MergeAIGatewayConfig().get_models_with_info(api_key="sk-merge")

        entry = catalog["anthropic/claude-3-haiku"]
        assert entry["input_cost_per_token"] == original["input_cost_per_token"]
        assert entry["output_cost_per_token"] == original["output_cost_per_token"]


class TestCatalogCache:
    def test_caches_within_ttl(self, respx_mock):
        route = respx_mock.get(OPENROUTER_URL).mock(return_value=httpx.Response(200, json={"data": [OPENROUTER_ITEM]}))

        first = get_catalog("openrouter", "sk-or", None)
        second = get_catalog("openrouter", "sk-or", None)

        assert len(route.calls) == 1
        assert first is second
        assert "anthropic/claude-sonnet-4" in first

    def test_cache_key_separates_api_base_and_key(self, respx_mock):
        respx_mock.route().mock(return_value=httpx.Response(200, json={"data": [OPENROUTER_ITEM]}))

        same_base = get_catalog("openrouter", "sk-or", None)
        other_base = get_catalog("openrouter", "sk-or", "https://other.example.com/v1")
        other_key = get_catalog("openrouter", "sk-other", None)

        assert len(gateway_catalog_cache._CATALOG_CACHE) == 3
        assert (same_base is other_base) is False
        assert (same_base is other_key) is False

    def test_ttl_expiry_refetches(self, respx_mock):
        route = respx_mock.get(OPENROUTER_URL).mock(return_value=httpx.Response(200, json={"data": [OPENROUTER_ITEM]}))

        get_catalog("openrouter", "sk-or", None)
        key = next(iter(gateway_catalog_cache._CATALOG_CACHE))
        ts, value = gateway_catalog_cache._CATALOG_CACHE[key]
        gateway_catalog_cache._CATALOG_CACHE[key] = (ts - 400, value)
        refetched = get_catalog("openrouter", "sk-or", None)

        assert len(route.calls) == 2
        assert "anthropic/claude-sonnet-4" in refetched

    def test_failed_fetch_returns_none_and_caches_nothing(self, respx_mock):
        respx_mock.get(OPENROUTER_URL).mock(return_value=httpx.Response(500, text="boom"))

        assert get_catalog("openrouter", "sk-or", None) is None
        assert gateway_catalog_cache._CATALOG_CACHE == {}

    def test_unknown_provider_returns_none(self):
        assert get_catalog("not_a_provider", None, None) is None

    def test_provider_without_catalog_returns_none(self):
        assert get_catalog("openai", "sk-openai", None) is None


class TestRegisterCatalog:
    def test_registers_prefixed_keys(self):
        key = "merge/anthropic/claude-opus-4-6"
        try:
            register_catalog_into_model_cost(
                "merge",
                MappingProxyType({"anthropic/claude-opus-4-6": {"input_cost_per_token": 5e-6, "mode": "chat"}}),
            )
            assert litellm.model_cost[key]["input_cost_per_token"] == 5e-6
        finally:
            litellm.model_cost.pop(key, None)
