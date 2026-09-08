import json
from datetime import datetime
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.token_counter import high_detail_image_token_upper_bound
from litellm.llms.azure.passthrough.transformation import AzurePassthroughConfig
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import EmbeddingResponse, ModelResponse


def _azure_chat_completion_body():
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4.1-mini-2025-04-14",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }


def _make_httpx_response(body: dict) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        content=json.dumps(body).encode("utf-8"),
        request=httpx.Request(
            "POST",
            "https://example.openai.azure.com/openai/deployments/gpt-4.1-mini/chat/completions",
        ),
    )


def test_azure_passthrough_logging_non_streaming_response_chat_completions():
    """
    Returns a populated ModelResponse (with usage + content) for a chat/completions
    endpoint. This is what _success_handler_helper_fn needs to build
    standard_logging_object — without it, Datadog/cost-tracking/router-success all
    raise on every Azure passthrough request.
    """
    config = AzurePassthroughConfig()
    logging_obj = MagicMock()

    result = config.logging_non_streaming_response(
        model="gpt-4.1-mini",
        custom_llm_provider="azure",
        httpx_response=_make_httpx_response(_azure_chat_completion_body()),
        request_data={
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}],
        },
        logging_obj=logging_obj,
        endpoint="openai/deployments/gpt-4.1-mini/chat/completions",
    )

    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "Hello! How can I assist you today?"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 8
    assert result.usage.total_tokens == 18


def _relay_logging_obj(model: str) -> Logging:
    logging_obj = Logging(
        model=model,
        messages=[],
        stream=False,
        call_type="allm_passthrough_route",
        start_time=datetime.now(),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    logging_obj.update_environment_variables(
        model=model,
        litellm_params={"api_base": "https://my-resource.openai.azure.com", "custom_llm_provider": "azure"},
        optional_params={},
        custom_llm_provider="azure",
    )
    return logging_obj


def _relay_logging_result(model: str, endpoint: str, body, status_code: int = 200):
    logging_obj = _relay_logging_obj(model)
    response = httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=json.dumps(body).encode("utf-8"),
        request=httpx.Request("POST", f"https://my-resource.openai.azure.com/{endpoint}?api-version=2025-04-01-preview"),
    )
    result = AzurePassthroughConfig().logging_non_streaming_response(
        model=model,
        custom_llm_provider="azure",
        httpx_response=response,
        request_data={},
        logging_obj=logging_obj,
        endpoint=endpoint,
    )
    return result, logging_obj


EMBEDDINGS_BODY = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 1000, "total_tokens": 1000},
}

RESPONSES_BODY = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-4.1-mini",
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
}


def test_azure_passthrough_embeddings_relay_is_costed_per_input_token():
    result, logging_obj = _relay_logging_result(
        "text-embedding-3-small", "openai/deployments/text-embedding-3-small/embeddings", EMBEDDINGS_BODY
    )
    per_token = litellm.get_model_info("azure/text-embedding-3-small")["input_cost_per_token"]

    assert isinstance(result, EmbeddingResponse)
    assert logging_obj.call_type == "aembedding"
    assert per_token > 0
    assert logging_obj._response_cost_calculator(result=result) == pytest.approx(1000 * per_token)


def test_azure_passthrough_responses_relay_is_costed_per_token():
    result, logging_obj = _relay_logging_result("gpt-4.1-mini", "openai/responses", RESPONSES_BODY)
    info = litellm.get_model_info("azure/gpt-4.1-mini")

    assert isinstance(result, ResponsesAPIResponse)
    assert logging_obj.call_type == "aresponses"
    assert logging_obj._response_cost_calculator(result=result) == pytest.approx(
        1000 * info["input_cost_per_token"] + 100 * info["output_cost_per_token"]
    )


def test_azure_passthrough_failed_embeddings_relay_is_not_costed():
    result, logging_obj = _relay_logging_result(
        "text-embedding-3-small",
        "openai/deployments/text-embedding-3-small/embeddings",
        {"error": {"code": "429", "message": "rate limited"}},
        status_code=429,
    )

    assert result is None
    assert logging_obj.call_type == "allm_passthrough_route"


def test_azure_passthrough_logging_non_streaming_response_unknown_endpoint_returns_none():
    result, logging_obj = _relay_logging_result(
        "gpt-4o-mini-tts", "openai/deployments/gpt-4o-mini-tts/audio/speech", {"audio": "..."}
    )

    assert result is None
    assert logging_obj.call_type == "allm_passthrough_route"


def _sse_line(payload: dict) -> str:
    return "data: " + json.dumps(payload)


def _azure_chat_completion_chunks() -> list[str]:
    head = {"id": "chatcmpl-abc123", "object": "chat.completion.chunk", "created": 1700000000, "model": "gpt-4.1-mini"}
    return [
        _sse_line(
            {
                **head,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello!"}, "finish_reason": None}],
            }
        ),
        _sse_line(
            {**head, "choices": [{"index": 0, "delta": {"content": " How can I assist?"}, "finish_reason": None}]}
        ),
        _sse_line({**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        _sse_line({**head, "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}}),
        "data: [DONE]",
    ]


def test_azure_passthrough_streaming_chat_chunks_build_the_complete_response():
    response = AzurePassthroughConfig().handle_logging_collected_chunks(
        all_chunks=_azure_chat_completion_chunks(),
        litellm_logging_obj=MagicMock(),
        model="gpt-4.1-mini",
        custom_llm_provider="azure",
        endpoint="openai/deployments/gpt-4.1-mini/chat/completions",
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Hello! How can I assist?"
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 8


def test_azure_passthrough_streaming_chunks_without_usage_count_prompt_tokens_from_the_relayed_request():
    messages = [{"role": "user", "content": "Say hi in three words"}]
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"request_data": {"messages": messages, "stream": True}}

    response = AzurePassthroughConfig().handle_logging_collected_chunks(
        all_chunks=[chunk for chunk in _azure_chat_completion_chunks() if '"usage"' not in chunk],
        litellm_logging_obj=logging_obj,
        model="gpt-4.1-mini",
        custom_llm_provider="azure",
        endpoint="openai/deployments/gpt-4.1-mini/chat/completions",
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Hello! How can I assist?"
    assert response.usage.prompt_tokens > 0
    assert response.usage.prompt_tokens == litellm.token_counter(model="gpt-4.1-mini", messages=messages)
    assert response.usage.completion_tokens > 0


def test_azure_passthrough_streaming_chunks_count_remote_image_prompt_tokens_without_fetching_the_image():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": "http://127.0.0.1:9/doc.png", "detail": "high"}},
            ],
        }
    ]
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"request_data": {"messages": messages, "stream": True}}

    response = AzurePassthroughConfig().handle_logging_collected_chunks(
        all_chunks=[chunk for chunk in _azure_chat_completion_chunks() if '"usage"' not in chunk],
        litellm_logging_obj=logging_obj,
        model="gpt-4.1-mini",
        custom_llm_provider="azure",
        endpoint="openai/deployments/gpt-4.1-mini/chat/completions",
    )

    text_only_messages = [{"role": "user", "content": [{"type": "text", "text": "Describe this"}]}]
    assert isinstance(response, ModelResponse)
    assert response.usage.prompt_tokens == (
        litellm.token_counter(model="gpt-4.1-mini", messages=text_only_messages) + high_detail_image_token_upper_bound()
    )


def test_azure_passthrough_streaming_chunks_for_unknown_endpoint_return_none():
    response = AzurePassthroughConfig().handle_logging_collected_chunks(
        all_chunks=_azure_chat_completion_chunks(),
        litellm_logging_obj=MagicMock(),
        model="gpt-4.1-mini",
        custom_llm_provider="azure",
        endpoint="openai/deployments/gpt-4.1-mini/embeddings",
    )

    assert response is None


def _complete_url(request_query_params: dict, litellm_params: dict) -> httpx.URL:
    url, _ = AzurePassthroughConfig().get_complete_url(
        api_base="https://my-resource.openai.azure.com",
        api_key="key",
        model="gpt-4.1-mini",
        endpoint="openai/deployments/gpt-4.1-mini/chat/completions",
        request_query_params=request_query_params,
        litellm_params=litellm_params,
    )
    return url


def test_azure_passthrough_url_forwards_the_callers_api_version():
    url = _complete_url(request_query_params={"api-version": "2025-04-01-preview"}, litellm_params={})

    assert url.path == "/openai/deployments/gpt-4.1-mini/chat/completions"
    assert url.params["api-version"] == "2025-04-01-preview"


def test_azure_passthrough_url_prefers_the_callers_api_version_over_the_deployments():
    url = _complete_url(
        request_query_params={"api-version": "2025-04-01-preview"}, litellm_params={"api_version": "2024-10-21"}
    )

    assert url.params["api-version"] == "2025-04-01-preview"


def test_azure_passthrough_url_fills_in_the_deployments_api_version_when_the_caller_sends_none():
    url = _complete_url(request_query_params={}, litellm_params={"api_version": "2024-10-21"})

    assert url.params["api-version"] == "2024-10-21"


def test_azure_passthrough_url_strips_the_leading_router_model_segment():
    url, _ = AzurePassthroughConfig().get_complete_url(
        api_base="https://my-resource.openai.azure.com",
        api_key="key",
        model="gpt-4.1-mini",
        endpoint="gpt-4.1-mini/openai/deployments/gpt-4.1-mini/chat/completions",
        request_query_params={"api-version": "2024-10-21"},
        litellm_params={},
    )

    assert str(url) == "https://my-resource.openai.azure.com/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2024-10-21"


def test_azure_passthrough_url_rewrites_the_model_group_only_as_a_whole_segment():
    url, _ = AzurePassthroughConfig().get_complete_url(
        api_base="https://my-resource.openai.azure.com",
        api_key="key",
        model="gpt-4.1-mini",
        endpoint="gpt/openai/deployments/gpt-4o/chat/completions",
        request_query_params={"api-version": "2024-10-21"},
        litellm_params={"litellm_metadata": {"model_group": "gpt"}},
    )

    assert str(url) == "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"


@pytest.mark.parametrize(
    "request_data, expected",
    [({"stream": True}, True), ({"stream": 1}, True), ({"stream": False}, False), ({}, False)],
)
def test_azure_passthrough_is_streaming_request_reads_the_stream_flag(request_data, expected):
    assert AzurePassthroughConfig().is_streaming_request(endpoint="openai/deployments/x/chat/completions", request_data=request_data) is expected
