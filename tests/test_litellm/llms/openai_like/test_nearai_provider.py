"""Tests for the JSON-configured NEAR AI Cloud provider."""

import io
import json
from pathlib import Path

import httpx
import litellm
import pytest
from openai import OpenAI

from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.rerank import RerankResponse


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
    assert "/v1/completions" in config.supported_endpoints
    assert "/v1/responses" in config.supported_endpoints

    model, provider, _, api_base = get_llm_provider("nearai/openai/gpt-oss-120b")
    assert model == "openai/gpt-oss-120b"
    assert provider == "nearai"
    assert api_base == NEARAI_API_BASE

    model, provider, dynamic_api_key, api_base = get_llm_provider(
        "Qwen/Qwen3.8-27B",
        api_base=NEARAI_API_BASE,
        api_key="near-test-key",
    )
    assert model == "Qwen/Qwen3.8-27B"
    assert provider == "nearai"
    assert dynamic_api_key == "near-test-key"
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

    embedding_config = ProviderConfigManager.get_provider_embedding_config(
        model="Qwen/Qwen3-Embedding-0.6B",
        provider=litellm.LlmProviders.NEARAI,
    )
    assert embedding_config is not None

    image_edit_config = ProviderConfigManager.get_provider_image_edit_config(
        model="black-forest-labs/FLUX.2-klein-4B",
        provider=litellm.LlmProviders.NEARAI,
    )
    assert image_edit_config is not None
    assert (
        image_edit_config.get_complete_url(
            model="black-forest-labs/FLUX.2-klein-4B",
            api_base=None,
            litellm_params={},
        )
        == f"{NEARAI_API_BASE}/images/edits"
    )


def test_nearai_text_completion_dispatches_to_completions_endpoint():
    def handle_request(request: httpx.Request) -> httpx.Response:
        assert request.url == f"{NEARAI_API_BASE}/completions"
        assert request.headers["authorization"] == "Bearer near-test-key"
        assert json.loads(request.content)["prompt"] == "hello"
        return httpx.Response(
            200,
            json={
                "id": "cmpl-near",
                "object": "text_completion",
                "created": 1,
                "model": "openai/gpt-oss-120b",
                "choices": [{"text": "world", "index": 0, "logprobs": None, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    openai_client = OpenAI(api_key="near-test-key", base_url=NEARAI_API_BASE, http_client=http_client)
    response = litellm.text_completion(
        model="nearai/openai/gpt-oss-120b",
        prompt="hello",
        api_key="near-test-key",
        client=openai_client,
    )
    assert response.choices[0].text == "world"


def test_nearai_embedding_and_image_edit_dispatch():
    def handle_request(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer near-test-key"
        if request.url.path == "/v1/embeddings":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
                    "model": "Qwen/Qwen3-Embedding-0.6B",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        assert request.url.path == "/v1/images/edits"
        return httpx.Response(200, json={"created": 1, "data": [{"b64_json": "aW1hZ2U="}]})

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    handler = HTTPHandler(client=http_client)
    embedding_response = litellm.embedding(
        model="nearai/Qwen/Qwen3-Embedding-0.6B",
        input=["hello"],
        api_key="near-test-key",
        client=handler,
    )
    assert embedding_response.data[0]["embedding"] == [0.1, 0.2]

    image_response = litellm.image_edit(
        model="nearai/black-forest-labs/FLUX.2-klein-4B",
        image=io.BytesIO(b"image"),
        prompt="edit",
        api_key="near-test-key",
        api_base=NEARAI_API_BASE,
        client=handler,
    )
    assert image_response.data[0]["b64_json"] == "aW1hZ2U="


def test_nearai_catalog_uses_current_specialized_pricing_fields(monkeypatch):
    repo_root = Path(__file__).resolve().parents[4]
    model_cost = json.loads((repo_root / "model_prices_and_context_window.json").read_text())
    backup_cost = json.loads((repo_root / "litellm/model_prices_and_context_window_backup.json").read_text())

    nearai_model_cost = {key: value for key, value in model_cost.items() if key.startswith("nearai/")}
    assert len(nearai_model_cost) == 48
    assert nearai_model_cost == {key: value for key, value in backup_cost.items() if key.startswith("nearai/")}

    reranker = model_cost["nearai/Qwen/Qwen3-Reranker-0.6B"]
    assert reranker["mode"] == "rerank"
    assert reranker["input_cost_per_query"] == 1e-8
    assert "input_cost_per_token" not in reranker

    image_model = model_cost["nearai/black-forest-labs/FLUX.2-klein-4B"]
    assert image_model["mode"] == "image_generation"
    assert image_model["input_cost_per_image"] == 0.012

    transcription_model = model_cost["nearai/openai/whisper-large-v3"]
    assert transcription_model["input_cost_per_second"] == 1e-8
    assert "input_cost_per_token" not in transcription_model

    monkeypatch.setitem(litellm.model_cost, "nearai/black-forest-labs/FLUX.2-klein-4B", image_model)
    monkeypatch.setitem(litellm.model_cost, "nearai/openai/whisper-large-v3", transcription_model)

    from litellm.cost_calculator import default_image_cost_calculator, transcription_cost

    assert default_image_cost_calculator(
        model="nearai/black-forest-labs/FLUX.2-klein-4B",
        custom_llm_provider="nearai",
        n=2,
    ) == pytest.approx(0.024)
    prompt_cost, completion_cost = transcription_cost(
        model="openai/whisper-large-v3",
        custom_llm_provider="nearai",
        duration=12,
    )
    assert prompt_cost == pytest.approx(1.2e-7)
    assert completion_cost == 0

    rerank_config = litellm.ProviderConfigManager.get_provider_rerank_config(
        model="Qwen/Qwen3-Reranker-0.6B",
        provider=litellm.LlmProviders.NEARAI,
        api_base=NEARAI_API_BASE,
        present_version_params=[],
    )
    rerank_response = rerank_config.transform_rerank_response(
        model="Qwen/Qwen3-Reranker-0.6B",
        raw_response=httpx.Response(
            200,
            json={
                "id": "rerank-near",
                "results": [{"index": 0, "relevance_score": 0.9}],
                "usage": {"total_tokens": 32},
            },
        ),
        model_response=RerankResponse(),
        logging_obj=None,
    )
    assert rerank_response.meta is not None
    billed_units = rerank_response.meta["billed_units"]
    assert billed_units is not None
    assert billed_units["search_units"] == 1
    prompt_cost, completion_cost = rerank_config.calculate_rerank_cost(
        model="Qwen/Qwen3-Reranker-0.6B",
        custom_llm_provider="nearai",
        billed_units=billed_units,
        model_info=reranker,
    )
    assert prompt_cost == pytest.approx(1e-8)
    assert completion_cost == 0

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
