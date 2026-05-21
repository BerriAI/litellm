"""Tests for the JSON-configured NEAR AI Cloud provider."""

import json
from pathlib import Path

import litellm


NEARAI_API_BASE = "https://cloud-api.near.ai/v1"


def test_nearai_json_registry_and_provider_resolution():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert litellm.LlmProviders.NEARAI.value == "nearai"
    assert "nearai" in litellm.provider_list
    assert JSONProviderRegistry.exists("nearai")

    config = JSONProviderRegistry.get("nearai")
    assert config is not None
    assert config.base_url == NEARAI_API_BASE
    assert config.api_key_env == "NEARAI_API_KEY"
    assert config.api_base_env == "NEARAI_API_BASE"
    assert config.param_mappings["max_completion_tokens"] == "max_tokens"
    assert "/v1/responses" in config.supported_endpoints

    model, provider, _, api_base = get_llm_provider("nearai/openai/gpt-oss-120b")
    assert model == "openai/gpt-oss-120b"
    assert provider == "nearai"
    assert api_base == NEARAI_API_BASE


def test_nearai_openai_compatible_endpoint_routing():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry
    from litellm.utils import ProviderConfigManager

    assert "nearai" in litellm.openai_compatible_providers
    assert "nearai" in litellm.openai_text_completion_compatible_providers

    provider = JSONProviderRegistry.get("nearai")
    assert provider is not None
    chat_config = create_config_class(provider)()
    assert (
        chat_config.get_complete_url(
            api_base=None,
            api_key=None,
            model="openai/gpt-oss-120b",
            optional_params={},
            litellm_params={},
        )
        == f"{NEARAI_API_BASE}/chat/completions"
    )

    responses_config = ProviderConfigManager.get_provider_responses_api_config(
        provider="nearai",
        model="openai/gpt-oss-120b",
    )
    assert responses_config is not None
    assert responses_config.get_complete_url(api_base=None, litellm_params={}) == f"{NEARAI_API_BASE}/responses"

    rerank_config = ProviderConfigManager.get_provider_rerank_config(
        model="Qwen/Qwen3-Reranker-0.6B",
        provider=litellm.LlmProviders.NEARAI,
        api_base=NEARAI_API_BASE,
        present_version_params=[],
    )
    assert (
        rerank_config.get_complete_url(
            api_base=NEARAI_API_BASE,
            model="Qwen/Qwen3-Reranker-0.6B",
        )
        == f"{NEARAI_API_BASE}/rerank"
    )


def test_nearai_catalog_uses_current_specialized_pricing_fields():
    repo_root = Path(__file__).resolve().parents[4]
    model_cost = json.loads((repo_root / "model_prices_and_context_window.json").read_text())

    reranker = model_cost["nearai/Qwen/Qwen3-Reranker-0.6B"]
    assert reranker["mode"] == "rerank"
    assert reranker["input_cost_per_query"] == 1e-8
    assert "input_cost_per_token" not in reranker

    image_model = model_cost["nearai/black-forest-labs/FLUX.2-klein-4B"]
    assert image_model["mode"] == "image_generation"
    assert image_model["output_cost_per_image"] == 0.012

    assert "nearai/Qwen/Qwen3-30B-A3B-Instruct-2507" not in model_cost
    assert "nearai/google/gemini-3-pro" not in model_cost
    assert "nearai/Qwen/Qwen3.8-27B" in model_cost
    assert "nearai/z-ai/glm-5.2" in model_cost


def test_nearai_supported_endpoint_matrices_match():
    repo_root = Path(__file__).resolve().parents[4]
    source = json.loads((repo_root / "provider_endpoints_support.json").read_text())
    backup = json.loads((repo_root / "litellm/provider_endpoints_support_backup.json").read_text())

    source_nearai = source["providers"]["nearai"]
    backup_nearai = backup["providers"]["nearai"]
    assert source_nearai == backup_nearai

    endpoints = source_nearai["endpoints"]
    assert endpoints["chat_completions"] is True
    assert endpoints["text_completion"] is True
    assert endpoints["responses"] is True
    assert endpoints["embeddings"] is True
    assert endpoints["image_generations"] is True
    assert endpoints["image_edits"] is True
    assert endpoints["audio_transcriptions"] is True
    assert endpoints["rerank"] is True
