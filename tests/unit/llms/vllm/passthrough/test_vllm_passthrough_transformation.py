import json
from datetime import datetime

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.vllm.passthrough.transformation import VLLMPassthroughConfig

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MESSAGES = [{"role": "user", "content": "Say hi in three words"}]
PROMPT = "The capital of France is"
USAGE = {"prompt_tokens": 34, "completion_tokens": 4, "total_tokens": 38}


def _relay_logging_obj(endpoint: str, request_data: dict) -> Logging:
    logging_obj = Logging(
        model=MODEL,
        messages=[],
        stream="stream" in request_data,
        call_type="allm_passthrough_route",
        start_time=datetime(2025, 1, 1),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    logging_obj.update_environment_variables(
        model=MODEL,
        litellm_params={"api_base": "http://vllm.local:8000/v1"},
        optional_params={},
        custom_llm_provider="hosted_vllm",
        endpoint=endpoint,
        request_data=request_data,
    )
    return logging_obj


def _relayed_json(path: str, body: dict) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        content=json.dumps(body).encode(),
        request=httpx.Request("POST", f"http://vllm.local:8000/{path}"),
    )


def _chat_completion(identity: str) -> httpx.Response:
    body = {
        "id": identity,
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello there!"}, "finish_reason": "stop"}],
        "usage": USAGE,
    }
    return _relayed_json("v1/chat/completions", body)


def _chat_stream(identity: str, usage: dict | None) -> bytes:
    head = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": MODEL}
    frames = [
        {**head, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}]},
        {**head, "choices": [{"index": 0, "delta": {"content": " there!"}, "finish_reason": "stop"}]},
        *([{**head, "choices": [], "usage": usage}] if usage else []),
    ]
    return b"".join(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames) + b"data: [DONE]\n\n"


def _text_stream(identity: str, deltas: list[tuple[int, str]], usage: dict | None) -> bytes:
    head = {"id": identity, "object": "text_completion", "created": 1, "model": MODEL}
    frames = [
        *({**head, "choices": [{"index": index, "text": text, "finish_reason": None}]} for index, text in deltas),
        *([{**head, "choices": [], "usage": usage}] if usage else []),
    ]
    return b"".join(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames) + b"data: [DONE]\n\n"


async def _log_stream(endpoint: str, sse: bytes, request_data: dict | None = None) -> Logging:
    logging_obj = _relay_logging_obj(endpoint, request_data or {"model": MODEL, "messages": MESSAGES, "stream": True})
    collector = VLLMPassthroughConfig().create_stream_collector(
        model=MODEL, custom_llm_provider="hosted_vllm", endpoint=endpoint
    )
    collector.add(sse[:57])
    collector.add(sse[57:])
    await logging_obj.async_flush_passthrough_collected_chunks(collector)
    return logging_obj


@pytest.mark.asyncio
async def test_a_relayed_chat_completion_is_logged_with_the_usage_vllm_reported():
    logging_obj = _relay_logging_obj("chat/completions", {"model": MODEL, "messages": MESSAGES})

    await logging_obj.async_success_handler(result=_chat_completion("chatcmpl-vllm-1"))

    payload = logging_obj.model_call_details["standard_logging_object"]
    assert payload["id"] == "chatcmpl-vllm-1"
    assert payload["custom_llm_provider"] == "hosted_vllm"
    assert (payload["prompt_tokens"], payload["completion_tokens"], payload["total_tokens"]) == (34, 4, 38)


@pytest.mark.asyncio
async def test_a_relayed_chat_stream_is_logged_with_the_usage_chunk_vllm_sent():
    logging_obj = await _log_stream("v1/chat/completions", _chat_stream("chatcmpl-vllm-2", USAGE))

    payload = logging_obj.model_call_details["standard_logging_object"]
    assert payload["id"] == "chatcmpl-vllm-2"
    assert payload["response"]["choices"][0]["message"]["content"] == "Hello there!"
    assert (payload["prompt_tokens"], payload["completion_tokens"]) == (34, 4)


@pytest.mark.asyncio
async def test_a_relayed_chat_stream_without_a_usage_chunk_counts_tokens_from_the_relayed_request():
    logging_obj = await _log_stream("chat/completions", _chat_stream("chatcmpl-vllm-3", None))

    payload = logging_obj.model_call_details["standard_logging_object"]
    assert payload["id"] == "chatcmpl-vllm-3"
    assert payload["prompt_tokens"] == litellm.token_counter(model=MODEL, messages=MESSAGES)
    assert payload["completion_tokens"] > 0


@pytest.mark.asyncio
async def test_a_relayed_text_completion_is_logged_with_the_usage_vllm_reported():
    logging_obj = _relay_logging_obj("v1/completions", {"model": MODEL, "prompt": PROMPT})
    body = {
        "id": "cmpl-vllm-4",
        "object": "text_completion",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "text": " Paris.", "finish_reason": "stop", "stop_reason": None}],
        "usage": USAGE,
    }

    result = VLLMPassthroughConfig().logging_non_streaming_response(
        model=MODEL,
        custom_llm_provider="hosted_vllm",
        httpx_response=_relayed_json("v1/completions", body),
        request_data={"model": MODEL, "prompt": PROMPT},
        logging_obj=logging_obj,
        endpoint="v1/completions",
    )
    await logging_obj.async_success_handler(result=result)

    payload = logging_obj.model_call_details["standard_logging_object"]
    assert payload["id"] == "cmpl-vllm-4"
    assert payload["response"]["choices"][0]["text"] == " Paris."
    assert (payload["prompt_tokens"], payload["completion_tokens"], payload["total_tokens"]) == (34, 4, 38)


@pytest.mark.asyncio
async def test_a_relayed_text_completion_stream_is_logged_with_the_usage_chunk_vllm_sent():
    sse = _text_stream("cmpl-vllm-5", [(0, " Par"), (0, "is.")], USAGE)

    logging_obj = await _log_stream("v1/completions", sse, {"model": MODEL, "prompt": PROMPT, "stream": True})

    payload = logging_obj.model_call_details["standard_logging_object"]
    assert payload["id"] == "cmpl-vllm-5"
    assert payload["response"]["choices"][0]["text"] == " Paris."
    assert (payload["prompt_tokens"], payload["completion_tokens"]) == (34, 4)


@pytest.mark.asyncio
async def test_a_relayed_text_completion_stream_without_a_usage_chunk_counts_every_prompt_and_choice():
    prompts = ["Hi", "Yo"]
    sse = _text_stream("cmpl-vllm-6", [(0, "Hello"), (1, "Hey"), (0, " there"), (1, " you")], None)

    logging_obj = await _log_stream("completions", sse, {"model": MODEL, "prompt": prompts, "stream": True})

    payload = logging_obj.model_call_details["standard_logging_object"]
    assert payload["id"] == "cmpl-vllm-6"
    assert [choice["text"] for choice in payload["response"]["choices"]] == ["Hello there", "Hey you"]
    assert payload["prompt_tokens"] == sum(litellm.token_counter(model=MODEL, text=prompt) for prompt in prompts)
    assert payload["completion_tokens"] == sum(
        litellm.token_counter(model=MODEL, text=text, count_response_tokens=True) for text in ["Hello there", "Hey you"]
    )
