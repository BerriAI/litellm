import json
from datetime import datetime
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.azure_ai.passthrough.transformation import AzureAIPassthroughConfig
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.types.rerank import RerankResponse
from litellm.types.utils import EmbeddingResponse, ImageResponse, LlmProviders, ModelResponse
from litellm.utils import ProviderConfigManager

FOUNDRY_BASE = "https://my-resource.services.ai.azure.com"
RESPONSES_COMPLETED_EVENT = {
    "type": "response.completed",
    "sequence_number": 2,
    "response": {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5.4-mini",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100},
    },
}


class _SpendProbe(CustomLogger):
    logged_call_type: str | None = None
    logged_cost: float | None = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.logged_call_type = kwargs["call_type"]
        self.logged_cost = kwargs["response_cost"]


@pytest.fixture(autouse=True)
def clear_azure_ai_env(monkeypatch):
    for env_var in ("AZURE_AI_API_BASE", "AZURE_AI_API_KEY", "AZURE_AD_TOKEN", "AZURE_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.setattr(litellm, "api_key", None)


def test_provider_config_manager_resolves_azure_ai_passthrough_config():
    config = ProviderConfigManager.get_provider_passthrough_config(
        model="Cohere-parse-v5", provider=LlmProviders.AZURE_AI
    )

    assert isinstance(config, AzureAIPassthroughConfig)


def test_router_model_prefix_is_stripped_and_native_path_kept_verbatim():
    url, base = AzureAIPassthroughConfig().get_complete_url(
        api_base=FOUNDRY_BASE,
        api_key=None,
        model="Cohere-parse-v5",
        endpoint="Cohere-parse-v5/providers/cohere/v2/parse",
        request_query_params=None,
        litellm_params={},
    )

    assert str(url) == f"{FOUNDRY_BASE}/providers/cohere/v2/parse"
    assert base == FOUNDRY_BASE


def test_model_group_prefix_is_stripped_when_router_metadata_names_it():
    url, _ = AzureAIPassthroughConfig().get_complete_url(
        api_base=FOUNDRY_BASE,
        api_key=None,
        model="Cohere-parse-v5",
        endpoint="/parse-alias/providers/cohere/v2/parse",
        request_query_params=None,
        litellm_params={"litellm_metadata": {"model_group": "parse-alias"}},
    )

    assert str(url) == f"{FOUNDRY_BASE}/providers/cohere/v2/parse"


def test_model_inside_the_path_stays_and_query_params_are_forwarded():
    url, _ = AzureAIPassthroughConfig().get_complete_url(
        api_base=f"{FOUNDRY_BASE}/",
        api_key=None,
        model="gpt-5.4-mini",
        endpoint="openai/deployments/gpt-5.4-mini/chat/completions",
        request_query_params={"api-version": "2024-10-21"},
        litellm_params={},
    )

    assert str(url) == f"{FOUNDRY_BASE}/openai/deployments/gpt-5.4-mini/chat/completions?api-version=2024-10-21"


def test_api_base_that_already_ends_in_models_is_cut_back_to_the_foundry_root():
    url, base = AzureAIPassthroughConfig().get_complete_url(
        api_base=f"{FOUNDRY_BASE}/models",
        api_key="key",
        model="gpt-5.4-mini",
        endpoint="gpt-5.4-mini/models/chat/completions",
        request_query_params={"api-version": "2024-05-01-preview"},
        litellm_params={},
    )

    assert str(url) == f"{FOUNDRY_BASE}/models/chat/completions?api-version=2024-05-01-preview"
    assert base == FOUNDRY_BASE


def test_full_url_api_base_that_already_ends_with_the_native_path_is_not_doubled():
    model_router_url = (
        "https://my-resource.cognitiveservices.azure.com/openai/deployments/model-router/chat/completions"
    )

    url, base = AzureAIPassthroughConfig().get_complete_url(
        api_base=f"{model_router_url}?api-version=2025-01-01-preview",
        api_key="key",
        model="model_router/model-router",
        endpoint="model-router/chat/completions",
        request_query_params=None,
        litellm_params={"litellm_metadata": {"model_group": "model-router"}},
    )

    assert str(url) == f"{model_router_url}?api-version=2025-01-01-preview"
    assert base == "https://my-resource.cognitiveservices.azure.com/openai/deployments/model-router"


@pytest.mark.parametrize("relayed_deployment", ["gpt-4o", "GPT-4o"])
def test_deployment_root_api_base_is_not_repeated_when_the_relay_carries_the_deployment_path(relayed_deployment):
    url, base = AzureAIPassthroughConfig().get_complete_url(
        api_base="https://my-resource.openai.azure.com/openai/deployments/gpt-4o",
        api_key="key",
        model="gpt-4o",
        endpoint=f"aoai-gpt-4o/openai/deployments/{relayed_deployment}/chat/completions",
        request_query_params={"api-version": "2024-10-21"},
        litellm_params={"litellm_metadata": {"model_group": "aoai-gpt-4o"}},
    )

    assert str(url) == (
        f"https://my-resource.openai.azure.com/openai/deployments/{relayed_deployment}/chat/completions"
        "?api-version=2024-10-21"
    )
    assert base == "https://my-resource.openai.azure.com"


def test_deployment_named_like_the_first_native_segment_keeps_its_deployment_root():
    url, base = AzureAIPassthroughConfig().get_complete_url(
        api_base="https://my-resource.openai.azure.com/openai/deployments/chat",
        api_key="key",
        model="chat",
        endpoint="aoai-chat/chat/completions",
        request_query_params={"api-version": "2024-10-21"},
        litellm_params={"litellm_metadata": {"model_group": "aoai-chat"}},
    )

    assert str(url) == "https://my-resource.openai.azure.com/openai/deployments/chat/chat/completions?api-version=2024-10-21"
    assert base == "https://my-resource.openai.azure.com/openai/deployments/chat"


def test_parse_relay_under_a_models_api_base_targets_the_foundry_root():
    url, _ = AzureAIPassthroughConfig().get_complete_url(
        api_base=f"{FOUNDRY_BASE}/models",
        api_key="key",
        model="Cohere-parse-v5",
        endpoint="Cohere-parse-v5/providers/cohere/v2/parse",
        request_query_params=None,
        litellm_params={},
    )

    assert str(url) == f"{FOUNDRY_BASE}/providers/cohere/v2/parse"


def test_deployment_api_version_fills_in_when_the_caller_sends_none():
    url, _ = AzureAIPassthroughConfig().get_complete_url(
        api_base=FOUNDRY_BASE,
        api_key="key",
        model="gpt-5.4-mini",
        endpoint="gpt-5.4-mini/models/chat/completions",
        request_query_params=None,
        litellm_params={"api_version": "2024-05-01-preview"},
    )

    assert str(url) == f"{FOUNDRY_BASE}/models/chat/completions?api-version=2024-05-01-preview"


def test_callers_api_version_beats_the_deployments():
    url, _ = AzureAIPassthroughConfig().get_complete_url(
        api_base=FOUNDRY_BASE,
        api_key="key",
        model="gpt-5.4-mini",
        endpoint="gpt-5.4-mini/models/chat/completions",
        request_query_params={"api-version": "2025-04-01-preview"},
        litellm_params={"api_version": "2024-05-01-preview"},
    )

    assert str(url) == f"{FOUNDRY_BASE}/models/chat/completions?api-version=2025-04-01-preview"


def test_api_version_on_the_configured_api_base_is_the_last_fallback():
    url, _ = AzureAIPassthroughConfig().get_complete_url(
        api_base=f"{FOUNDRY_BASE}/models/chat/completions?api-version=2024-05-01-preview",
        api_key="key",
        model="gpt-5.4-mini",
        endpoint="gpt-5.4-mini/models/chat/completions",
        request_query_params=None,
        litellm_params={},
    )

    assert str(url) == f"{FOUNDRY_BASE}/models/chat/completions?api-version=2024-05-01-preview"


def test_missing_api_base_raises_instead_of_building_a_relative_url():
    with pytest.raises(ValueError, match="AZURE_AI_API_BASE"):
        AzureAIPassthroughConfig().get_complete_url(
            api_base=None,
            api_key=None,
            model="Cohere-parse-v5",
            endpoint="Cohere-parse-v5/providers/cohere/v2/parse",
            request_query_params=None,
            litellm_params={},
        )


def _auth_headers(api_key: str | None, api_base: str, litellm_params: dict | None = None) -> dict:
    return AzureAIPassthroughConfig().validate_environment(
        headers={"content-type": "application/json"},
        model="Cohere-parse-v5",
        messages=[],
        optional_params={},
        litellm_params=litellm_params or {},
        api_key=api_key,
        api_base=api_base,
    )


def test_foundry_host_gets_the_api_key_header():
    headers = _auth_headers(api_key="deployment-key", api_base=FOUNDRY_BASE)

    assert headers == {"content-type": "application/json", "api-key": "deployment-key"}


def test_serverless_host_gets_a_bearer_token():
    headers = _auth_headers(api_key="deployment-key", api_base="https://cohere-parse.eastus.models.ai.azure.com")

    assert headers["Authorization"] == "Bearer deployment-key"
    assert "api-key" not in headers


def test_entra_token_is_used_when_the_deployment_has_no_api_key():
    headers = _auth_headers(api_key=None, api_base=FOUNDRY_BASE, litellm_params={"azure_ad_token": "entra-token"})

    assert headers["Authorization"] == "Bearer entra-token"


def test_no_credentials_at_all_raises():
    with pytest.raises(ValueError, match="Missing Azure AI credentials"):
        _auth_headers(api_key=None, api_base=FOUNDRY_BASE)


@pytest.mark.parametrize(
    "request_data, expected",
    [({"stream": True}, True), ({"stream": 1}, True), ({"stream": False}, False), ({}, False)],
)
def test_is_streaming_request_reads_the_stream_flag(request_data, expected):
    assert (
        AzureAIPassthroughConfig().is_streaming_request(endpoint="models/chat/completions", request_data=request_data)
        is expected
    )


def _chat_completion_response() -> httpx.Response:
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-5.4-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }
    return httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        content=json.dumps(body).encode("utf-8"),
        request=httpx.Request("POST", f"{FOUNDRY_BASE}/models/chat/completions"),
    )


def test_chat_completions_relay_yields_a_model_response_for_cost_tracking():
    result = AzureAIPassthroughConfig().logging_non_streaming_response(
        model="gpt-5.4-mini",
        custom_llm_provider="azure_ai",
        httpx_response=_chat_completion_response(),
        request_data={"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "hi"}]},
        logging_obj=MagicMock(),
        endpoint="models/chat/completions",
    )

    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "hi"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 8


def _non_chat_logging_result(content: bytes, content_type: str):
    parse_response = httpx.Response(
        status_code=200,
        headers={"content-type": content_type},
        content=content,
        request=httpx.Request("POST", f"{FOUNDRY_BASE}/providers/cohere/v2/parse"),
    )
    return AzureAIPassthroughConfig().logging_non_streaming_response(
        model="Cohere-parse-v5",
        custom_llm_provider="azure_ai",
        httpx_response=parse_response,
        request_data={"model": "Cohere-parse-v5"},
        logging_obj=MagicMock(),
        endpoint="providers/cohere/v2/parse",
    )


def test_non_chat_relay_with_a_non_json_body_logs_the_raw_text():
    assert _non_chat_logging_result(b"page one", "text/plain") == {"response": "page one"}


def _relay_logging_obj(
    model: str,
    api_base: str,
    stream: bool = False,
    callbacks: list[CustomLogger] | None = None,
    endpoint: str = "",
) -> Logging:
    logging_obj = Logging(
        model=model,
        messages=[],
        stream=stream,
        call_type="allm_passthrough_route",
        start_time=datetime.now(),
        litellm_call_id="call-1",
        function_id="fn-1",
        dynamic_async_success_callbacks=callbacks,
    )
    logging_obj.update_environment_variables(
        model=model,
        litellm_params={"api_base": api_base, "custom_llm_provider": "azure_ai"},
        optional_params={},
        custom_llm_provider="azure_ai",
        endpoint=endpoint,
    )
    return logging_obj


def _relay_logging_result(
    config: AzureAIPassthroughConfig,
    model: str,
    native_path: str,
    body,
    api_base: str = FOUNDRY_BASE,
    status_code: int = 200,
):
    relayed_url = f"{FOUNDRY_BASE}/{native_path}?api-version=2024-05-01-preview"
    logging_obj = _relay_logging_obj(model, api_base)
    response = httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=json.dumps(body).encode("utf-8"),
        request=httpx.Request("POST", relayed_url),
    )
    result = config.logging_non_streaming_response(
        model=model,
        custom_llm_provider="azure_ai",
        httpx_response=response,
        request_data={"model": model},
        logging_obj=logging_obj,
        endpoint=f"{model}/{native_path}",
    )
    return result, logging_obj


MISTRAL_OCR_BODY = {
    "pages": [{"index": 0, "markdown": "page one"}, {"index": 1, "markdown": "page two"}],
    "model": "mistral-document-ai-2512",
    "usage_info": {"pages_processed": 2, "doc_size_bytes": 4321},
}


def test_mistral_document_ai_relay_is_costed_per_page():
    result, logging_obj = _relay_logging_result(
        AzureAIPassthroughConfig(), "mistral-document-ai-2512", "providers/mistral/azure/ocr", MISTRAL_OCR_BODY
    )
    per_page = litellm.get_model_info("azure_ai/mistral-document-ai-2512")["ocr_cost_per_page"]

    assert isinstance(result, OCRResponse)
    assert result.usage_info.pages_processed == 2
    assert per_page > 0
    assert logging_obj._response_cost_calculator(result=result) == pytest.approx(2 * per_page)


def test_ocr_route_under_a_models_api_base_is_still_recognised():
    result, _ = _relay_logging_result(
        AzureAIPassthroughConfig(),
        "mistral-document-ai-2512",
        "providers/mistral/azure/ocr",
        MISTRAL_OCR_BODY,
        api_base=f"{FOUNDRY_BASE}/models",
    )

    assert isinstance(result, OCRResponse)


def test_relay_to_a_non_ocr_route_keeps_the_passthrough_object_and_call_type():
    result, logging_obj = _relay_logging_result(
        AzureAIPassthroughConfig(), "mistral-document-ai-2512", "models/info", {"name": "mistral-document-ai-2512"}
    )

    assert result == {"response": {"name": "mistral-document-ai-2512"}}
    assert logging_obj.call_type == "allm_passthrough_route"


COHERE_PARSE_BODY = {"id": "parse-1", "pages": [], "meta": {"billed_units": {"pages": 3}}}


def test_cohere_parse_relay_is_costed_per_billed_page():
    result, logging_obj = _relay_logging_result(
        AzureAIPassthroughConfig(), "Cohere-parse-v5", "providers/cohere/v2/parse", COHERE_PARSE_BODY
    )
    per_page = litellm.get_model_info("azure_ai/Cohere-parse-v5")["ocr_cost_per_page"]

    assert isinstance(result, OCRResponse)
    assert result.usage_info.pages_processed == 3
    assert logging_obj.call_type == "aocr"
    assert per_page > 0
    assert logging_obj._response_cost_calculator(result=result) == pytest.approx(3 * per_page)


def test_deployment_without_an_ocr_config_is_never_costed_as_ocr():
    config = AzureAIPassthroughConfig(ocr_config_for=lambda model: None)
    result, logging_obj = _relay_logging_result(
        config, "mistral-document-ai-2512", "providers/mistral/azure/ocr", MISTRAL_OCR_BODY
    )

    assert result == {"response": MISTRAL_OCR_BODY}
    assert logging_obj.call_type == "allm_passthrough_route"


def test_accepted_ocr_job_without_a_result_body_is_not_costed():
    result, logging_obj = _relay_logging_result(
        AzureAIPassthroughConfig(),
        "mistral-document-ai-2512",
        "providers/mistral/azure/ocr",
        {"status": "running"},
        status_code=202,
    )

    assert result == {"response": {"status": "running"}}
    assert logging_obj.call_type == "allm_passthrough_route"


def test_unparseable_ocr_body_falls_back_to_the_passthrough_object():
    result, logging_obj = _relay_logging_result(
        AzureAIPassthroughConfig(),
        "mistral-document-ai-2512",
        "providers/mistral/azure/ocr",
        ["not", "an", "ocr", "body"],
    )

    assert result == {"response": '["not", "an", "ocr", "body"]'}
    assert logging_obj.call_type == "allm_passthrough_route"


EMBEDDINGS_BODY = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    "model": "embed-v-4-0",
    "usage": {"prompt_tokens": 1200, "total_tokens": 1200},
}

RERANK_BODY = {
    "id": "rerank-1",
    "results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}],
    "meta": {"api_version": {"version": "2"}, "billed_units": {"search_units": 2}},
}

IMAGE_BODY = {"created": 1, "data": [{"b64_json": "AAAA"}]}


def test_foundry_embeddings_relay_is_costed_per_input_token():
    result, logging_obj = _relay_logging_result(
        AzureAIPassthroughConfig(), "embed-v-4-0", "models/embeddings", EMBEDDINGS_BODY
    )
    per_token = litellm.get_model_info("azure_ai/embed-v-4-0")["input_cost_per_token"]

    assert isinstance(result, EmbeddingResponse)
    assert logging_obj.call_type == "aembedding"
    assert per_token > 0
    assert logging_obj._response_cost_calculator(result=result) == pytest.approx(1200 * per_token)


def test_cohere_rerank_relay_is_costed_per_search_unit():
    result, logging_obj = _relay_logging_result(
        AzureAIPassthroughConfig(), "cohere-rerank-v4.0-fast", "providers/cohere/v2/rerank", RERANK_BODY
    )
    per_query = litellm.get_model_info("azure_ai/cohere-rerank-v4.0-fast")["input_cost_per_query"]

    assert isinstance(result, RerankResponse)
    assert logging_obj.call_type == "arerank"
    assert per_query > 0
    assert logging_obj._response_cost_calculator(result=result) == pytest.approx(2 * per_query)


def test_image_generation_relay_is_costed_per_image():
    result, logging_obj = _relay_logging_result(
        AzureAIPassthroughConfig(), "FLUX.2-pro", "openai/deployments/FLUX.2-pro/images/generations", IMAGE_BODY
    )
    per_image = litellm.get_model_info("azure_ai/FLUX.2-pro")["output_cost_per_image"]

    assert isinstance(result, ImageResponse)
    assert logging_obj.call_type == "aimage_generation"
    assert per_image > 0
    assert logging_obj._response_cost_calculator(result=result) == pytest.approx(per_image)


def test_flux_2_relay_through_the_provider_route_is_costed_per_image():
    result, logging_obj = _relay_logging_result(
        AzureAIPassthroughConfig(), "FLUX.2-pro", "providers/blackforestlabs/v1/flux-2-pro", IMAGE_BODY
    )
    per_image = litellm.get_model_info("azure_ai/FLUX.2-pro")["output_cost_per_image"]

    assert isinstance(result, ImageResponse)
    assert logging_obj.call_type == "aimage_generation"
    assert logging_obj._response_cost_calculator(result=result) == pytest.approx(per_image)


def test_rejected_rerank_relay_keeps_the_passthrough_object_and_call_type():
    result, logging_obj = _relay_logging_result(
        AzureAIPassthroughConfig(),
        "cohere-rerank-v4.0-fast",
        "providers/cohere/v2/rerank",
        {"message": "invalid request"},
        status_code=400,
    )

    assert result == {"response": {"message": "invalid request"}}
    assert logging_obj.call_type == "allm_passthrough_route"


def test_streaming_chat_completion_chunks_are_costed_like_azure():
    head = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-5.4-mini"}
    chunks = [
        "data: "
        + json.dumps(
            {
                **head,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            }
        ),
        "data: "
        + json.dumps({**head, "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}),
        "data: [DONE]",
    ]

    response = AzureAIPassthroughConfig().handle_logging_collected_chunks(
        all_chunks=chunks,
        litellm_logging_obj=MagicMock(),
        model="gpt-5.4-mini",
        custom_llm_provider="azure_ai",
        endpoint="chat/completions",
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "hi"
    assert response.usage.total_tokens == 4


def test_streaming_responses_chunks_through_a_router_relay_are_costed_like_azure():
    logging_obj = _relay_logging_obj("gpt-5.4-mini", FOUNDRY_BASE)

    response = AzureAIPassthroughConfig().handle_logging_collected_chunks(
        all_chunks=["event: response.completed", "data: " + json.dumps(RESPONSES_COMPLETED_EVENT)],
        litellm_logging_obj=logging_obj,
        model="gpt-5.4-mini",
        custom_llm_provider="azure_ai",
        endpoint="gpt/openai/responses",
    )
    info = litellm.get_model_info("azure_ai/gpt-5.4-mini")

    assert response is not None
    assert response.response.usage.output_tokens == 100
    assert logging_obj.call_type == "aresponses"
    assert logging_obj._response_cost_calculator(result=response.response) == pytest.approx(
        1000 * info["input_cost_per_token"] + 100 * info["output_cost_per_token"]
    )


async def test_streaming_responses_relay_flush_reaches_the_success_callbacks_with_a_price():
    probe = _SpendProbe()
    logging_obj = _relay_logging_obj(
        "gpt-5.4-mini", FOUNDRY_BASE, stream=True, callbacks=[probe], endpoint="gpt/openai/responses"
    )
    stream = "event: response.completed\ndata: " + json.dumps(RESPONSES_COMPLETED_EVENT) + "\n\n"

    await logging_obj.async_flush_passthrough_collected_chunks(
        raw_bytes=[stream.encode()], provider_config=AzureAIPassthroughConfig()
    )
    info = litellm.get_model_info("azure_ai/gpt-5.4-mini")

    assert probe.logged_call_type == "allm_passthrough_route"
    assert probe.logged_cost == pytest.approx(1000 * info["input_cost_per_token"] + 100 * info["output_cost_per_token"])
