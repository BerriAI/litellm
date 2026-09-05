import json
from unittest.mock import MagicMock

import httpx
import pytest

import litellm


from litellm.litellm_core_utils.token_counter import high_detail_image_token_upper_bound
from litellm.llms.azure.passthrough.transformation import AzurePassthroughConfig
from litellm.types.utils import ModelResponse


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


def test_azure_passthrough_logging_non_streaming_response_unknown_endpoint_returns_none():
    """
    Endpoints other than chat/completions (responses, messages, images) fall
    through to None — matches base-class behavior and Bedrock's "unknown
    endpoint" handling. Not a regression; just scoping.
    """
    config = AzurePassthroughConfig()
    logging_obj = MagicMock()

    result = config.logging_non_streaming_response(
        model="gpt-4.1-mini",
        custom_llm_provider="azure",
        httpx_response=_make_httpx_response(_azure_chat_completion_body()),
        request_data={},
        logging_obj=logging_obj,
        endpoint="openai/responses",
    )

    assert result is None


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


def test_azure_passthrough_url_falls_back_to_the_callers_api_version():
    url = _complete_url(request_query_params={"api-version": "2025-04-01-preview"}, litellm_params={})

    assert url.path == "/openai/deployments/gpt-4.1-mini/chat/completions"
    assert url.params["api-version"] == "2025-04-01-preview"


def test_azure_passthrough_url_prefers_the_deployments_api_version():
    url = _complete_url(
        request_query_params={"api-version": "2025-04-01-preview"}, litellm_params={"api_version": "2024-10-21"}
    )

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
