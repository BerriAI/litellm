"""
Unit tests for xAI Grok models on Vertex AI partner model
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../../../"))  # repo root

# Force litellm to load model_cost from the local backup JSON
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm
from litellm.llms.vertex_ai.vertex_ai_partner_models.main import (
    PartnerModelPrefixes,
    VertexAIPartnerModels,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.llms.vertex_ai import VertexPartnerProvider


# Helpers
XAI_MODELS = [
    "xai/grok-4.1-fast-non-reasoning",
    "xai/grok-4.1-fast-reasoning",
    "xai/grok-4.20-non-reasoning",
    "xai/grok-4.20-reasoning",
]

NON_XAI_MODELS = [
    "meta/llama-3.1-8b-instruct-maas",
    "claude-3-5-sonnet-v2@20241022",
    "mistral-large@2407",
]


# 1. PartnerModelPrefixes enum
def test_xai_prefix_exists_in_enum():
    """XAI_PREFIX must exist and equal 'xai/'"""
    assert hasattr(PartnerModelPrefixes, "XAI_PREFIX")
    assert PartnerModelPrefixes.XAI_PREFIX == "xai/"


# 2. is_vertex_partner_model()
@pytest.mark.parametrize("model", XAI_MODELS)
def test_xai_is_recognised_as_partner_model(model):
    """is_vertex_partner_model must return True for all xai/grok-* models"""
    assert VertexAIPartnerModels.is_vertex_partner_model(model) is True


@pytest.mark.parametrize("model", NON_XAI_MODELS)
def test_non_xai_partner_models_unaffected(model):
    """Non-xAI partner models must still be recognised correctly"""
    assert VertexAIPartnerModels.is_vertex_partner_model(model) is True


# 3. should_use_openai_handler()
@pytest.mark.parametrize("model", XAI_MODELS)
def test_xai_uses_openai_handler(model):
    """xAI Grok models must be routed through the OpenAI-compatible handler"""
    assert VertexAIPartnerModels.should_use_openai_handler(model) is True


def test_llama_still_uses_openai_handler():
    """llama models must still use the OpenAI handler"""
    assert (
        VertexAIPartnerModels.should_use_openai_handler(
            "meta/llama-3.1-8b-instruct-maas"
        )
        is True
    )


# 4. VertexPartnerProvider enum
def test_vertex_partner_provider_has_xai():
    """VertexPartnerProvider enum must contain an xai member"""
    assert hasattr(VertexPartnerProvider, "xai")
    assert VertexPartnerProvider.xai == "xai"


# 5. create_vertex_url()
def test_xai_url_uses_openapi_endpoint():
    """
    create_vertex_url must return the /endpoints/openapi/chat/completions path
    for VertexPartnerProvider.xai identical to how llama models are handled
    """
    url = VertexBase.create_vertex_url(
        vertex_location="us-central1",
        vertex_project="my-project",
        partner=VertexPartnerProvider.xai,
        stream=False,
        model="xai/grok-4.1-fast-non-reasoning",
    )
    assert "/endpoints/openapi/chat/completions" in url
    assert "us-central1" in url
    assert "my-project" in url


def test_xai_streaming_url_same_as_non_streaming():
    """
    The openapi endpoint does not have a separate streaming URL;
    stream=True must produce the same base path
    """
    url_sync = VertexBase.create_vertex_url(
        vertex_location="us-central1",
        vertex_project="my-project",
        partner=VertexPartnerProvider.xai,
        stream=False,
        model="xai/grok-4.1-fast-reasoning",
    )
    url_stream = VertexBase.create_vertex_url(
        vertex_location="us-central1",
        vertex_project="my-project",
        partner=VertexPartnerProvider.xai,
        stream=True,
        model="xai/grok-4.1-fast-reasoning",
    )
    assert "/endpoints/openapi/chat/completions" in url_sync
    assert "/endpoints/openapi/chat/completions" in url_stream


def test_llama_url_unaffected():
    """llama URL must still use /endpoints/openapi/chat/completions"""
    url = VertexBase.create_vertex_url(
        vertex_location="us-central1",
        vertex_project="my-project",
        partner=VertexPartnerProvider.llama,
        stream=False,
        model="meta/llama-3.1-8b-instruct-maas",
    )
    assert "/endpoints/openapi/chat/completions" in url


# 6. Pricing entries in model_prices_and_context_window.json
EXPECTED_PRICING_KEYS = [
    "vertex_ai/xai/grok-4.1-fast-non-reasoning",
    "vertex_ai/xai/grok-4.1-fast-reasoning",
    "vertex_ai/xai/grok-4.20-non-reasoning",
    "vertex_ai/xai/grok-4.20-reasoning",
]


@pytest.mark.parametrize("model_key", EXPECTED_PRICING_KEYS)
def test_xai_model_in_pricing_json(model_key):
    """All xAI Grok Vertex models must have pricing entries"""
    assert (
        model_key in litellm.model_cost
    ), f"'{model_key}' not found in model_prices_and_context_window.json"


@pytest.mark.parametrize("model_key", EXPECTED_PRICING_KEYS)
def test_xai_pricing_has_required_fields(model_key):
    """Each xAI pricing entry must have provider, mode, cost, and context tokens"""
    entry = litellm.model_cost[model_key]
    assert entry.get("litellm_provider") == "vertex_ai"
    assert entry.get("mode") == "chat"
    assert "input_cost_per_token" in entry
    assert "output_cost_per_token" in entry
    assert "max_input_tokens" in entry
    assert "max_output_tokens" in entry


def test_reasoning_models_have_reasoning_flag():
    """Models labelled 'reasoning' must set supports_reasoning=true"""
    for key in EXPECTED_PRICING_KEYS:
        if "reasoning" in key and "non-reasoning" not in key:
            assert (
                litellm.model_cost[key].get("supports_reasoning") is True
            ), f"'{key}' should have supports_reasoning=true"


def test_non_reasoning_models_have_reasoning_false():
    """Models labelled 'non-reasoning' must NOT set supports_reasoning=true"""
    for key in EXPECTED_PRICING_KEYS:
        if "non-reasoning" in key:
            assert (
                litellm.model_cost[key].get("supports_reasoning") is not True
            ), f"'{key}' should NOT have supports_reasoning=true"


# 8. partner model config dispatch (__init__.py)
def test_get_vertex_ai_partner_model_config_for_xai():
    """
    get_vertex_ai_partner_model_config must return VertexAILlama3Config
    when vertex_publisher_or_api_spec == 'xai'
    """
    from litellm.llms.vertex_ai.vertex_ai_partner_models import (
        get_vertex_ai_partner_model_config,
    )
    from litellm.llms.vertex_ai.vertex_ai_partner_models.llama3.transformation import (
        VertexAILlama3Config,
    )

    config = get_vertex_ai_partner_model_config(
        model="xai/grok-4.1-fast-non-reasoning",
        vertex_publisher_or_api_spec="xai",
    )
    assert isinstance(config, VertexAILlama3Config)


def test_get_vertex_ai_partner_model_config_for_openapi():
    """Regression – 'openapi' spec must still return VertexAILlama3Config"""
    from litellm.llms.vertex_ai.vertex_ai_partner_models import (
        get_vertex_ai_partner_model_config,
    )
    from litellm.llms.vertex_ai.vertex_ai_partner_models.llama3.transformation import (
        VertexAILlama3Config,
    )

    config = get_vertex_ai_partner_model_config(
        model="meta/llama-3.1-8b-instruct-maas",
        vertex_publisher_or_api_spec="openapi",
    )
    assert isinstance(config, VertexAILlama3Config)


# 9. Token counter – publisher detection
@pytest.mark.parametrize("model", XAI_MODELS)
def test_token_counter_publisher_for_xai(model):
    """_get_publisher_for_model must return 'xai' for all xai/grok-* models"""
    from litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler import (
        VertexAIPartnerModelsTokenCounter,
    )

    counter = VertexAIPartnerModelsTokenCounter()
    result = counter._get_publisher_for_model(model)
    assert result == "xai"


def test_token_counter_publisher_regression_llama():
    """llama models must still return 'meta'."""
    from litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler import (
        VertexAIPartnerModelsTokenCounter,
    )

    counter = VertexAIPartnerModelsTokenCounter()
    assert counter._get_publisher_for_model("meta/llama-3.1-8b-instruct-maas") == "meta"
