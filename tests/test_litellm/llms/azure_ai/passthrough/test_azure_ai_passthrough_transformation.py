import json
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.azure_ai.passthrough.transformation import AzureAIPassthroughConfig
from litellm.types.utils import LlmProviders, ModelResponse
from litellm.utils import ProviderConfigManager

FOUNDRY_BASE = "https://my-resource.services.ai.azure.com"


@pytest.fixture(autouse=True)
def clear_azure_ai_env(monkeypatch):
    for env_var in ("AZURE_AI_API_BASE", "AZURE_AI_API_KEY", "AZURE_AD_TOKEN", "AZURE_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.setattr(litellm, "api_key", None)


def test_provider_config_manager_resolves_azure_ai_passthrough_config():
    config = ProviderConfigManager.get_provider_passthrough_config(model="Cohere-parse-v5", provider=LlmProviders.AZURE_AI)

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
    assert AzureAIPassthroughConfig().is_streaming_request(endpoint="models/chat/completions", request_data=request_data) is expected


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


def test_non_chat_relay_logs_the_parsed_body_so_spend_tracking_sees_the_call():
    result = _non_chat_logging_result(b'{"id":"parse-1","pages":[],"meta":{"billed_units":{"pages":1}}}', "application/json")

    assert result == {"response": {"id": "parse-1", "pages": [], "meta": {"billed_units": {"pages": 1}}}}


def test_non_chat_relay_with_a_non_json_body_logs_the_raw_text():
    assert _non_chat_logging_result(b"page one", "text/plain") == {"response": "page one"}


def test_streaming_chat_completion_chunks_are_costed_like_azure():
    head = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-5.4-mini"}
    chunks = [
        "data: " + json.dumps({**head, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}),
        "data: " + json.dumps({**head, "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}),
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
