from datetime import datetime
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)
from litellm.types.utils import ModelResponse, PassthroughCallTypes

_OMNI_INTERACTIONS_USAGE: Final = {
    "total_tokens": 4041,
    "total_input_tokens": 12,
    "input_tokens_by_modality": [{"modality": "text", "tokens": 12}],
    "total_output_tokens": 4009,
    "output_tokens_by_modality": [
        {"modality": "text", "tokens": 9},
        {"modality": "video", "tokens": 4000},
    ],
    "total_tool_use_tokens": 0,
    "total_thought_tokens": 20,
}


def test_interactions_create_response_logs_modality_usage_and_cost() -> None:
    """
    Regression for LIT-6896: gemini-omni Interactions passthrough rows were logged
    with zero tokens and zero spend. Input, text-output and video-output tokens
    must land in usage, priced with the model's per-modality rates, and the
    response id must stay the litellm_call_id so SpendLogs keep their request_id.
    """
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj.optional_params = {}
    logging_obj.litellm_call_id = "call-6896"
    response = httpx.Response(
        status_code=200,
        json={
            "id": "interactions/abc",
            "model": "gemini-omni-flash-preview",
            "status": "completed",
            "outputs": [{"type": "text", "text": "hi"}],
            "usage": _OMNI_INTERACTIONS_USAGE,
        },
    )

    result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
        httpx_response=response,
        logging_obj=logging_obj,
        url_route="https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/interactions",
        result=response.text,
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        request_body={"model": "gemini-omni-flash-preview", "input": [{"type": "text", "text": "say hi"}]},
    )

    model_response = result["result"]
    assert isinstance(model_response, ModelResponse)
    assert model_response.id == "call-6896"
    usage = model_response.usage
    assert usage.prompt_tokens == 12
    assert usage.completion_tokens == 4009 + 20
    assert usage.completion_tokens_details.text_tokens == 9
    assert usage.completion_tokens_details.video_tokens == 4000

    model_info = litellm.get_model_info(model="gemini-omni-flash-preview", custom_llm_provider="vertex_ai")
    expected_cost = (
        12 * model_info["input_cost_per_token"]
        + (9 + 20) * model_info["output_cost_per_token"]
        + 4000 * model_info["output_cost_per_video_token"]
    )
    assert result["kwargs"]["response_cost"] == pytest.approx(expected_cost)
    assert result["kwargs"]["custom_llm_provider"] == "vertex_ai"
    assert logging_obj.model_call_details["model"] == "gemini-omni-flash-preview"
    assert logging_obj.model_call_details["custom_llm_provider"] == "vertex_ai"


def test_interactions_response_without_usage_falls_back_to_generic_logging() -> None:
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj.optional_params = {}
    response = httpx.Response(status_code=200, json={"id": "interactions/abc", "status": "in_progress"})

    result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
        httpx_response=response,
        logging_obj=logging_obj,
        url_route="https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/interactions",
        result=response.text,
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        request_body={"agent": "projects/p/locations/global/reasoningEngines/1"},
    )

    assert result["result"] is None
    assert "response_cost" not in result["kwargs"]


def test_lyria_predict_response_preserves_audio_response_and_logs_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "vertex_ai/lyria-002",
        {
            "vertex_ai_audio_api": "lyria_predict",
            "supported_audio_formats": ["wav"],
            "output_cost_per_image": 0.06,
        },
    )
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    response = httpx.Response(
        status_code=200,
        json={
            "predictions": [
                {
                    "audioContent": "clip-1",
                    "mimeType": "audio/wav",
                },
                {
                    "audioContent": "clip-2",
                    "mimeType": "audio/wav",
                },
            ]
        },
    )

    result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
        httpx_response=response,
        logging_obj=logging_obj,
        url_route="/v1/projects/test/locations/us-central1/publishers/google/models/lyria-002:predict",
        result=response.text,
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        request_body={"instances": [{"prompt": "ambient piano"}]},
    )

    assert result["result"] == {
        "response": {
            "predictions": [
                {
                    "audioContent": "clip-1",
                    "mimeType": "audio/wav",
                },
                {
                    "audioContent": "clip-2",
                    "mimeType": "audio/wav",
                },
            ]
        }
    }
    assert result["kwargs"]["model"] == "lyria-002"
    assert result["kwargs"]["custom_llm_provider"] == "vertex_ai"
    assert result["kwargs"]["response_cost"] == pytest.approx(0.12)
    assert logging_obj.model == "lyria-002"
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(0.12)


def test_audio_predict_response_uses_model_map_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "vertex_ai/music-audio-preview",
        {
            "vertex_ai_audio_api": "lyria_predict",
            "supported_audio_formats": ["wav"],
            "output_cost_per_image": 0.5,
        },
    )
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    response = httpx.Response(
        status_code=200,
        json={
            "predictions": [
                {
                    "audioContent": "clip",
                    "mimeType": "audio/wav",
                }
            ]
        },
    )

    result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
        httpx_response=response,
        logging_obj=logging_obj,
        url_route="/v1/projects/test/locations/us-central1/publishers/google/models/music-audio-preview:predict",
        result=response.text,
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        request_body={"instances": [{"prompt": "ambient piano"}]},
    )

    assert result["kwargs"]["model"] == "music-audio-preview"
    assert result["kwargs"]["response_cost"] == pytest.approx(0.5)
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(0.5)


def test_audio_predict_response_supports_bytes_base64_encoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "vertex_ai/lyria-002",
        {
            "vertex_ai_audio_api": "lyria_predict",
            "supported_audio_formats": ["wav"],
            "output_cost_per_image": 0.06,
        },
    )
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    response = httpx.Response(
        status_code=200,
        json={"predictions": [{"bytesBase64Encoded": "clip"}]},
    )

    result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
        httpx_response=response,
        logging_obj=logging_obj,
        url_route="/v1/projects/test/locations/us-central1/publishers/google/models/lyria-002:predict",
        result=response.text,
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        request_body={"instances": [{"prompt": "ambient piano"}]},
    )

    assert result["kwargs"]["response_cost"] == pytest.approx(0.06)
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(0.06)


@pytest.mark.parametrize("runtime_entry_is_missing", (True, False))
def test_lyria_predict_cost_falls_back_to_bundled_map_when_runtime_metadata_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    runtime_entry_is_missing: bool,
    local_model_cost_map: None,
) -> None:
    if runtime_entry_is_missing:
        monkeypatch.delitem(litellm.model_cost, "vertex_ai/lyria-002")
    else:
        monkeypatch.setitem(
            litellm.model_cost,
            "vertex_ai/lyria-002",
            {
                key: value
                for key, value in litellm.model_cost["vertex_ai/lyria-002"].items()
                if key != "output_cost_per_image"
            },
        )
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    response = httpx.Response(
        status_code=200,
        json={
            "predictions": [
                {
                    "audioContent": "clip",
                    "mimeType": "audio/wav",
                }
            ]
        },
    )

    result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
        httpx_response=response,
        logging_obj=logging_obj,
        url_route="/v1/projects/test/locations/us-central1/publishers/google/models/lyria-002:predict",
        result=response.text,
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        request_body={"instances": [{"prompt": "ambient piano"}]},
    )

    if runtime_entry_is_missing:
        assert "vertex_ai/lyria-002" not in litellm.model_cost
    assert result["kwargs"]["model"] == "lyria-002"
    assert result["kwargs"]["response_cost"] == pytest.approx(0.06)
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(0.06)


def test_image_predict_response_is_not_billed_as_audio(
    local_model_cost_map: None,
) -> None:
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    response = httpx.Response(
        status_code=200,
        json={"predictions": [{"bytesBase64Encoded": "frame", "mimeType": "image/png"}]},
    )

    result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
        httpx_response=response,
        logging_obj=logging_obj,
        url_route=(
            "/v1/projects/test/locations/us-central1/publishers/google/models/imagen-4.0-generate-001:predict"
        ),
        result=response.text,
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        request_body={"instances": [{"prompt": "a red cube"}]},
    )

    assert isinstance(result["result"], litellm.ImageResponse)
    assert logging_obj.call_type == PassthroughCallTypes.passthrough_image_generation.value
    assert result["kwargs"]["response_cost"] == pytest.approx(
        litellm.model_cost["vertex_ai/imagen-4.0-generate-001"]["output_cost_per_image"]
    )
