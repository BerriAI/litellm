from typing import Optional
from unittest.mock import MagicMock

from httpx import Response

from litellm.llms.vllm.passthrough.transformation import VLLMPassthroughConfig
from litellm.types.utils import CostResponseTypes, EmbeddingResponse, ModelResponse


def _run_response(
    response: Response, endpoint: str = "/vllm/pooling"
) -> Optional[CostResponseTypes]:
    return VLLMPassthroughConfig().logging_non_streaming_response(
        model="intfloat/e5-mistral-7b-instruct",
        custom_llm_provider="vllm",
        httpx_response=response,
        request_data={},
        logging_obj=MagicMock(),
        endpoint=endpoint,
    )


def _run(
    response_json: object, endpoint: str = "/vllm/pooling"
) -> Optional[CostResponseTypes]:
    return _run_response(Response(status_code=200, json=response_json), endpoint)


def test_usage_bearing_response_is_cost_trackable():
    result = _run(
        {
            "model": "intfloat/e5-mistral-7b-instruct",
            "data": [{"index": 0, "data": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 11, "total_tokens": 11},
        }
    )
    assert isinstance(result, EmbeddingResponse)
    assert result.model == "intfloat/e5-mistral-7b-instruct"
    assert result.usage.prompt_tokens == 11
    assert result.usage.total_tokens == 11


def test_missing_total_tokens_defaults_to_prompt_tokens():
    result = _run(
        {
            "model": "intfloat/e5-mistral-7b-instruct",
            "data": [{"index": 0, "data": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 11},
        }
    )
    assert isinstance(result, EmbeddingResponse)
    assert result.usage.prompt_tokens == 11
    assert result.usage.total_tokens == 11


def test_chat_completion_preserves_completion_tokens():
    result = _run(
        {
            "id": "chatcmpl-vllm",
            "created": 1,
            "model": "example-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        },
        endpoint="/vllm/v1/chat/completions",
    )
    assert isinstance(result, ModelResponse)
    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 7
    assert result.usage.total_tokens == 18


def test_unsupported_endpoint_returns_none():
    assert (
        _run(
            {"usage": {"prompt_tokens": 11, "total_tokens": 11}},
            endpoint="/vllm/models",
        )
        is None
    )


def test_response_without_usage_returns_none():
    assert _run({"model": "m", "data": []}) is None


def test_non_dict_response_returns_none():
    assert _run(["not", "a", "dict"]) is None


def test_unparseable_body_returns_none():
    assert _run_response(Response(status_code=200, content=b"not json")) is None
