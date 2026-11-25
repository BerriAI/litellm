import base64
import io
import json
import wave
from datetime import datetime
from typing import Final

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)
from litellm.types.utils import EmbeddingResponse, ModelResponse

_PREDICT_ROUTE = "/v1/projects/p/locations/us-central1/publishers/google/models/text-embedding-004:predict"
_INTERACTIONS_ROUTE = "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/interactions"


def _handle(url_route: str, payload: object) -> PassThroughEndpointLoggingTypedDict:
    logging_obj = Logging(
        model="unknown",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    logging_obj.optional_params = {}
    response = httpx.Response(200, json=payload)
    return VertexPassthroughLoggingHandler.vertex_passthrough_handler(
        httpx_response=response,
        logging_obj=logging_obj,
        url_route=url_route,
        result=response.text,
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1),
        cache_hit=False,
        request_body={"model": "gemini-omni-flash-preview"},
    )


def test_predict_response_with_text_embeddings_is_logged_as_an_embedding_response():
    result = _handle(
        _PREDICT_ROUTE,
        {
            "predictions": [
                {"embeddings": {"values": [0.1, 0.2], "statistics": {"token_count": 3}}},
                {"embeddings": {"values": [0.3, 0.4], "statistics": {"token_count": 4}}},
            ],
            "metadata": {"billableCharacterCount": 9},
        },
    )

    response = result["result"]
    assert isinstance(response, EmbeddingResponse)
    assert response.data == [
        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
        {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
    ]
    assert response.usage.prompt_tokens == 7
    assert result["kwargs"]["model"] == "text-embedding-004"
    assert result["kwargs"]["custom_llm_provider"] == "vertex_ai"


@pytest.mark.parametrize("payload", [["not", "an", "object"], "text", 7, 1.5, True])
def test_predict_response_that_is_not_a_json_object_is_rejected_without_echoing_it(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        _handle(_PREDICT_ROUTE, payload)

    assert "input_value" not in str(exc_info.value)


def test_interactions_usage_object_is_read_into_prompt_and_completion_tokens():
    result = _handle(
        _INTERACTIONS_ROUTE,
        {
            "id": "interactions/abc",
            "model": "gemini-omni-flash-preview",
            "usage": {
                "total_tokens": 41,
                "total_input_tokens": 12,
                "input_tokens_by_modality": [{"modality": "text", "tokens": 12}],
                "total_output_tokens": 9,
                "output_tokens_by_modality": [{"modality": "text", "tokens": 9}],
                "total_thought_tokens": 20,
            },
        },
    )

    response = result["result"]
    assert isinstance(response, ModelResponse)
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 29
    assert response.usage.completion_tokens_details.text_tokens == 9
    assert result["kwargs"]["custom_llm_provider"] == "vertex_ai"


def test_build_complete_streaming_response_assembles_stream_generate_content_sse_chunks():
    logging_obj: Final = Logging(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    logging_obj.update_environment_variables(litellm_params={}, optional_params={}, model="gemini-2.5-flash")
    first_chunk: Final = {"candidates": [{"content": {"parts": [{"text": "Hello"}], "role": "model"}, "index": 0}]}
    last_chunk: Final = {
        "candidates": [
            {"content": {"parts": [{"text": " there!"}], "role": "model"}, "finishReason": "STOP", "index": 0}
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 8, "totalTokenCount": 18},
    }

    response: Final = VertexPassthroughLoggingHandler._build_complete_streaming_response(
        all_chunks=[f"data: {json.dumps(first_chunk)}", f"data: {json.dumps(last_chunk)}"],
        litellm_logging_obj=logging_obj,
        model="gemini-2.5-flash",
        url_route="/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash:streamGenerateContent",
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Hello there!"
    assert response.choices[0].finish_reason == "stop"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens) == (10, 8, 18)


@pytest.mark.parametrize(
    "undecodable_line",
    ["data: 5", "data: " + '{"a": ' * 100_000 + "1" + "}" * 100_000],
    ids=["number", "deep-object"],
)
def test_build_complete_streaming_response_skips_a_claude_data_line_that_is_not_an_object(undecodable_line: str):
    logging_obj: Final = Logging(
        model="claude-sonnet-5-5",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    logging_obj.update_environment_variables(litellm_params={}, optional_params={}, model="claude-sonnet-5-5")
    events: Final = (
        {"type": "message_start", "message": {"id": "msg_1", "model": "claude-sonnet-5-5", "usage": {"input_tokens": 5}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Paris"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    )
    first_line, *other_lines = (f"data: {json.dumps(event)}" for event in events)

    response: Final = VertexPassthroughLoggingHandler._build_complete_streaming_response(
        all_chunks=[first_line, undecodable_line, *other_lines],
        litellm_logging_obj=logging_obj,
        model="claude-sonnet-5-5",
        url_route="/v1/projects/p/locations/global/publishers/anthropic/models/claude-sonnet-5-5:streamRawPredict",
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Paris"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (5, 1)


def test_gemini_batch_embed_contents_passthrough_bills_image_tokens_from_usage_metadata():
    logging_obj: Final = Logging(
        model="unknown",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    logging_obj.optional_params = {}
    response: Final = httpx.Response(
        200,
        json={
            "embeddings": [{"values": [0.1, 0.2]}],
            "usageMetadata": {
                "promptTokenCount": 258,
                "promptTokenDetails": [{"modality": "IMAGE", "tokenCount": 258}],
            },
        },
    )

    result: Final = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
        httpx_response=response,
        logging_obj=logging_obj,
        url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents",
        result=response.text,
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1),
        cache_hit=False,
        request_body={
            "requests": [
                {
                    "model": "models/gemini-embedding-2",
                    "content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}]},
                }
            ]
        },
    )

    assert result["result"].usage.prompt_tokens == 258
    assert result["result"].usage.prompt_tokens_details.image_tokens == 258
    assert result["kwargs"]["response_cost"] == pytest.approx(
        258 * litellm.model_cost["gemini/gemini-embedding-2"]["input_cost_per_image_token"]
    )


def _gemini_tts_wav_base64() -> str:
    buffer: Final = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(bytes(range(256)) * 8)
    return base64.b64encode(buffer.getvalue()).decode()


def _gemini_tts_reply(wav_base64: str) -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {"parts": [{"inlineData": {"mimeType": "audio/wav", "data": wav_base64}}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 40, "totalTokenCount": 47},
        "modelVersion": "gemini-3.8-flash-tts",
    }


def test_vertex_passthrough_logs_the_tts_audio_the_caller_received() -> None:
    wav_base64: Final = _gemini_tts_wav_base64()
    reply: Final = _gemini_tts_reply(wav_base64)
    url_route: Final = "/v1/projects/p/locations/global/publishers/google/models/gemini-3.8-flash-tts:generateContent"
    logging_obj: Final = Logging(
        model="unknown",
        messages=[{"role": "user", "content": "Say hello"}],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    logging_obj.update_environment_variables(litellm_params={}, optional_params={}, model="gemini-3.8-flash-tts")

    logged: Final = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
        httpx_response=httpx.Response(200, json=reply),
        logging_obj=logging_obj,
        url_route=url_route,
        result=json.dumps(reply),
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1),
        cache_hit=False,
        request_body={},
    )["result"]

    assert isinstance(logged, ModelResponse)
    assert logged.choices[0].message.audio.data == wav_base64
