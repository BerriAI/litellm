import json
import os
import sys
import time
from unittest.mock import MagicMock

import httpx

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.llms.vllm.passthrough.transformation import VLLMPassthroughConfig
from litellm.types.utils import EmbeddingResponse, ModelResponse


def _vllm_pooling_body():
    return {
        "id": "pool-abc123",
        "object": "list",
        "created": 1700000000,
        "model": "BAAI/bge-m3",
        "data": [{"index": 0, "object": "pooling", "data": [0.1, 0.2, 0.3]}],
        "usage": {"prompt_tokens": 8, "total_tokens": 8, "completion_tokens": 0},
    }


def _vllm_chat_completion_body():
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "BAAI/bge-m3",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello there"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _make_httpx_response(body: dict, url: str) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        content=json.dumps(body).encode("utf-8"),
        request=httpx.Request("POST", url),
    )


def test_vllm_passthrough_logging_non_streaming_pooling_returns_usage():
    """
    A /pooling response (no chat choices) must still produce a usage-bearing
    response so _success_handler_helper_fn can build a standard_logging_object.
    Before this, VLLMPassthroughConfig fell through to the base no-op and every
    router-model /vllm/pooling request skipped LiteLLM_SpendLogs.
    """
    config = VLLMPassthroughConfig()

    result = config.logging_non_streaming_response(
        model="BAAI/bge-m3",
        custom_llm_provider="vllm",
        httpx_response=_make_httpx_response(
            _vllm_pooling_body(), "http://vllm:8000/pooling"
        ),
        request_data={"model": "BAAI/bge-m3", "input": "hi"},
        logging_obj=MagicMock(),
        endpoint="pooling",
    )

    assert isinstance(result, EmbeddingResponse)
    assert result.model == "BAAI/bge-m3"
    assert result.usage.prompt_tokens == 8
    assert result.usage.total_tokens == 8


def test_vllm_passthrough_logging_non_streaming_chat_completions_returns_model_response():
    config = VLLMPassthroughConfig()

    result = config.logging_non_streaming_response(
        model="BAAI/bge-m3",
        custom_llm_provider="vllm",
        httpx_response=_make_httpx_response(
            _vllm_chat_completion_body(), "http://vllm:8000/v1/chat/completions"
        ),
        request_data={"model": "BAAI/bge-m3", "messages": [{"role": "user", "content": "hi"}]},
        logging_obj=MagicMock(),
        endpoint="v1/chat/completions",
    )

    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "Hello there"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15


def test_vllm_passthrough_logging_non_streaming_no_usage_returns_none():
    config = VLLMPassthroughConfig()

    result = config.logging_non_streaming_response(
        model="BAAI/bge-m3",
        custom_llm_provider="vllm",
        httpx_response=_make_httpx_response(
            {"object": "list", "data": []}, "http://vllm:8000/pooling"
        ),
        request_data={},
        logging_obj=MagicMock(),
        endpoint="pooling",
    )

    assert result is None


def test_vllm_passthrough_non_streaming_builds_standard_logging_object():
    """
    End-to-end guard: the async success handler must build a
    standard_logging_object for a non-streaming /vllm/pooling router-model
    response. Reverting logging_non_streaming_response leaves it None, which is
    exactly what caused '_PROXY_track_cost_callback' and Prometheus to raise and
    the SpendLogs row to be skipped.
    """
    logging_obj = LitellmLogging(
        model="BAAI/bge-m3",
        messages=[],
        stream=False,
        call_type="allm_passthrough_route",
        start_time=time.time(),
        litellm_call_id="vllm-passthrough-1",
        function_id="1",
    )
    logging_obj.model_call_details["custom_llm_provider"] = "vllm"
    logging_obj.model_call_details["endpoint"] = "pooling"
    logging_obj.model_call_details["litellm_params"] = {"allm_passthrough_route": True}
    logging_obj.litellm_params = logging_obj.model_call_details["litellm_params"]

    httpx_response = _make_httpx_response(
        _vllm_pooling_body(), "http://vllm:8000/pooling"
    )

    logging_obj._success_handler_helper_fn(result=httpx_response)

    sl_object = logging_obj.model_call_details.get("standard_logging_object")
    assert sl_object is not None
    assert sl_object["response_cost"] is not None
    assert sl_object["total_tokens"] == 8
