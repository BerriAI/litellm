"""Regression tests for pass-through generic cost attribution.

Covers the seams where a pass-through call was either misclassified (wrong
handler → crash or $0 row) or mispriced (cached tokens billed flat, GET echoes
re-billed, provider fallback pricing a self-hosted model at the real
provider's rates).
"""

import os
import sys
import types
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm  # noqa: E402
from litellm.proxy.pass_through_endpoints.llm_provider_handlers import (  # noqa: E402
    openai_passthrough_logging_handler as openai_handler_module,
)
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (  # noqa: E402
    OpenAIPassthroughLoggingHandler,
    _is_responses_path,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (  # noqa: E402
    HttpPassThroughEndpointHelpers,
)
from litellm.proxy.pass_through_endpoints.success_handler import (  # noqa: E402
    PassThroughEndpointLogging,
    _resolve_generic_price,
    extract_generic_usage,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (  # noqa: E402
    EndpointType,
)

# ---------------------------------------------------------------------------
# is_cohere_route: host-gated. Path containment alone is a trap — "/v1/embed"
# is a substring of "/v1/embeddings", and the Cohere branch runs BEFORE the
# OpenAI one, so every OpenAI-shaped embeddings call would crash the
# spend-logging path in the Cohere transform.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://api.cohere.com/v1/embed", True),
        ("https://api.cohere.com/v2/chat", True),
        ("https://api.cohere.ai/v1/embed", True),
        ("https://api.openai.com/v1/embeddings", False),
        ("https://api.fireworks.ai/inference/v1/embeddings", False),
        ("https://my-resource.openai.azure.com/openai/v1/embeddings", False),
    ],
)
def test_is_cohere_route_is_host_gated(url, expected):
    handler = PassThroughEndpointLogging()
    assert bool(handler.is_cohere_route(url)) is expected


# ---------------------------------------------------------------------------
# extract_generic_usage: (prompt, completion, cache_read, cache_creation)
# ---------------------------------------------------------------------------


def test_extract_openai_shape_with_cached_tokens_details():
    usage = extract_generic_usage(
        {"usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 30}}}
    )
    assert usage == (100, 20, 30, 0)


def test_extract_deepseek_prompt_cache_hit_tokens():
    usage = extract_generic_usage(
        {"usage": {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_cache_hit_tokens": 800}}
    )
    assert usage == (1000, 100, 800, 0)


def test_extract_anthropic_shape_adds_cache_tokens_into_prompt():
    # Anthropic reports cache tokens OUTSIDE input_tokens; they must be folded
    # in so `prompt` means the same thing across shapes.
    usage = extract_generic_usage(
        {
            "usage": {
                "input_tokens": 50,
                "output_tokens": 10,
                "cache_read_input_tokens": 40,
                "cache_creation_input_tokens": 25,
            }
        }
    )
    assert usage == (115, 10, 40, 25)


def test_extract_gemini_shape_cached_content_and_thoughts():
    # cachedContentTokenCount is inside promptTokenCount; thoughtsTokenCount is
    # OUTSIDE candidatesTokenCount but billed at the output rate.
    usage = extract_generic_usage(
        {
            "usageMetadata": {
                "promptTokenCount": 200,
                "candidatesTokenCount": 30,
                "cachedContentTokenCount": 120,
                "thoughtsTokenCount": 70,
            }
        }
    )
    assert usage == (200, 100, 120, 0)


def test_extract_unrecognised_shape_returns_none():
    assert extract_generic_usage({"tokens": {"in": 5, "out": 5}}) is None


# ---------------------------------------------------------------------------
# _resolve_generic_price: no provider=None fallback when a provider is set.
# ---------------------------------------------------------------------------


def _fake_get_model_info(priced_providers_rates):
    """get_model_info stub keyed on custom_llm_provider; raises when unknown."""

    def _fake(model=None, custom_llm_provider=None):
        if custom_llm_provider in priced_providers_rates:
            return priced_providers_rates[custom_llm_provider]
        raise ValueError(f"model {model!r} isn't mapped for provider {custom_llm_provider!r}")

    return _fake


def test_resolve_generic_price_no_fallback_to_providerless_lookup(monkeypatch):
    # A vllm upstream serving "gpt-4o" must stay unpriced — not billed at
    # OpenAI's rates via a provider-less fallback lookup.
    rates = {None: {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}}
    monkeypatch.setattr("litellm.utils.get_model_info", _fake_get_model_info(rates))
    assert _resolve_generic_price(model="gpt-4o", custom_llm_provider="vllm") is None


def test_resolve_generic_price_returns_four_tuple(monkeypatch):
    rates = {
        "deepseek": {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "cache_read_input_token_cost": 1e-7,
            "cache_creation_input_token_cost": None,
        }
    }
    monkeypatch.setattr("litellm.utils.get_model_info", _fake_get_model_info(rates))
    assert _resolve_generic_price(model="deepseek-chat", custom_llm_provider="deepseek") == (
        1e-6,
        2e-6,
        1e-7,
        None,
    )


# ---------------------------------------------------------------------------
# _price_generic_passthrough end-to-end: cached tokens priced at the
# discounted rate; a cache component WITHOUT an explicit rate leaves the call
# unpriced instead of confidently overcharging.
# ---------------------------------------------------------------------------

_DEEPSEEK_USAGE_BODY = {
    "model": "deepseek-fake-chat",
    "usage": {"prompt_tokens": 1000, "prompt_cache_hit_tokens": 800, "completion_tokens": 100},
}


def _price_deepseek_call(monkeypatch, price_entry):
    monkeypatch.setattr("litellm.utils.get_model_info", _fake_get_model_info({"deepseek": price_entry}))
    logging_obj = types.SimpleNamespace(model_call_details={})
    handler = PassThroughEndpointLogging()
    kwargs = handler._price_generic_passthrough(
        response_body=_DEEPSEEK_USAGE_BODY,
        request_body={},
        logging_obj=logging_obj,
        url_route="https://self-hosted.example.com/chat/completions",
        custom_llm_provider="deepseek",
        kwargs={},
    )
    return kwargs, logging_obj


def test_generic_pricing_applies_cache_read_discount(monkeypatch):
    kwargs, logging_obj = _price_deepseek_call(
        monkeypatch,
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "cache_read_input_token_cost": 1e-7,
        },
    )
    # 200 uncached * 1e-6 + 800 cached * 1e-7 + 100 completion * 2e-6
    assert kwargs["response_cost"] == pytest.approx(0.00048)
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(0.00048)


def test_generic_pricing_leaves_call_unpriced_without_cache_read_rate(monkeypatch):
    kwargs, logging_obj = _price_deepseek_call(
        monkeypatch,
        {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6},
    )
    assert "response_cost" not in kwargs
    assert "response_cost" not in logging_obj.model_call_details


# ---------------------------------------------------------------------------
# openai_passthrough_handler: GET echoes must not be re-billed.
# ---------------------------------------------------------------------------

_CHAT_URL = "https://api.openai.com/v1/chat/completions"

_CHAT_BODY_WITH_USAGE = {
    "model": "gpt-4o",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _call_openai_handler(monkeypatch, method):
    # Isolate the method gate: the surface builder and standard-logging step
    # are stubbed so the test can't be perturbed by their internals.
    monkeypatch.setattr(
        openai_handler_module,
        "_build_response_and_cost_for_surface",
        lambda **kw: (litellm.ModelResponse(), 0.123),
    )
    monkeypatch.setattr(openai_handler_module, "get_standard_logging_object_payload", lambda **kw: None)

    httpx_response = httpx.Response(
        status_code=200,
        json=_CHAT_BODY_WITH_USAGE,
        request=httpx.Request(method, _CHAT_URL),
    )
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    return OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
        httpx_response=httpx_response,
        response_body=_CHAT_BODY_WITH_USAGE,
        logging_obj=logging_obj,
        url_route=_CHAT_URL,
        result="",
        start_time=None,
        end_time=None,
        cache_hit=False,
        request_body={"model": "gpt-4o"},
        litellm_params={},
    )


def test_openai_handler_get_echo_produces_no_cost(monkeypatch):
    # GET /v1/chat/completions lists stored completions; its body ECHOES the
    # original usage block. Costing it re-bills the generation on every poll.
    result = _call_openai_handler(monkeypatch, "GET")
    assert result["result"] is None
    assert "response_cost" not in result["kwargs"]


def test_openai_handler_post_still_prices(monkeypatch):
    result = _call_openai_handler(monkeypatch, "POST")
    assert result["result"] is not None
    assert result["kwargs"]["response_cost"] == pytest.approx(0.123)


# ---------------------------------------------------------------------------
# Route predicates: collection-only Responses path, final-segment embeddings.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/v1/responses", True),
        ("/openai/responses", True),
        ("/responses", True),
        ("/v1/responses/resp_123", False),
        ("/v1/responses/resp_123/cancel", False),
        ("/v1/responses/resp_123/input_items", False),
    ],
)
def test_is_responses_path_collection_only(path, expected):
    assert _is_responses_path(path) is expected


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://api.openai.com/v1/embeddings", True),
        ("https://api.openai.com/v1/embeddings/jobs", False),
        ("https://my-resource.openai.azure.com/openai/deployments/d1/embeddings", True),
    ],
)
def test_is_openai_embeddings_route_requires_final_segment(url, expected):
    assert OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(url) is expected


# ---------------------------------------------------------------------------
# get_endpoint_type: provider-keyed OpenAI-compatible upstreams classify
# OPENAI for streaming; without a provider they stay GENERIC.
# ---------------------------------------------------------------------------

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


def test_get_endpoint_type_provider_keyed_openai_compatible():
    assert (
        HttpPassThroughEndpointHelpers.get_endpoint_type(_GROQ_CHAT_URL, custom_llm_provider="groq")
        == EndpointType.OPENAI
    )


def test_get_endpoint_type_no_provider_stays_generic():
    assert HttpPassThroughEndpointHelpers.get_endpoint_type(_GROQ_CHAT_URL) == EndpointType.GENERIC


@pytest.mark.parametrize(
    "url,provider,expected",
    [
        ("https://api.anthropic.com/v1/messages", None, EndpointType.ANTHROPIC),
        ("https://api.cohere.com/v2/chat", None, EndpointType.COHERE),
        (
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/google/models/gemini:generateContent",
            None,
            EndpointType.VERTEX_AI,
        ),
    ],
)
def test_get_endpoint_type_other_branches_unaffected(url, provider, expected):
    assert HttpPassThroughEndpointHelpers.get_endpoint_type(url, custom_llm_provider=provider) == expected
