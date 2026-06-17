"""Unit tests for LyriaConfig.transform_response"""

import base64
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.vertex_ai.lyria.transformation import LyriaConfig, LyriaError
from litellm.types.utils import ChatCompletionAudioResponse, ModelResponse


def _make_raw_response(payload: dict) -> MagicMock:
    """Build a mock httpx.Response that returns payload from .json()."""
    mock = MagicMock()
    mock.json.return_value = payload
    mock.text = json.dumps(payload)
    return mock


def _make_model_response() -> ModelResponse:
    return ModelResponse()


SAMPLE_AUDIO_B64 = base64.b64encode(b"fake-mp3-bytes").decode()

SAMPLE_PAYLOAD = {
    "id": "uD8sarr0MZ6BvPgP_LDMyQI",
    "status": "completed",
    "role": "model",
    "object": "interaction",
    "model": "lyria-3-pro-preview",
    "outputs": [
        {"type": "text", "text": "[0.0:] verse one lyrics..."},
        {"type": "text", "text": "Caption: driving rock anthem"},
        {"type": "audio", "mime_type": "audio/mpeg", "data": SAMPLE_AUDIO_B64},
    ],
}


@pytest.fixture
def lyria_config() -> LyriaConfig:
    return LyriaConfig()


def _call_transform(lyria_config: LyriaConfig, payload: dict) -> ModelResponse:
    return lyria_config.transform_response(
        model="vertex_ai/lyria-3-pro-preview",
        raw_response=_make_raw_response(payload),
        model_response=_make_model_response(),
        logging_obj=MagicMock(),
        request_data={},
        messages=[{"role": "user", "content": "make music"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def test_audio_data_in_message_audio(lyria_config):
    result = _call_transform(lyria_config, SAMPLE_PAYLOAD)
    audio = result.choices[0].message.audio
    assert audio is not None
    assert audio.data == SAMPLE_AUDIO_B64


def test_lyrics_in_transcript(lyria_config):
    result = _call_transform(lyria_config, SAMPLE_PAYLOAD)
    audio = result.choices[0].message.audio
    assert audio.transcript == "[0.0:] verse one lyrics..."


def test_caption_in_provider_specific_fields(lyria_config):
    result = _call_transform(lyria_config, SAMPLE_PAYLOAD)
    psf = result.choices[0].message.provider_specific_fields
    assert psf is not None
    assert psf["caption"] == "Caption: driving rock anthem"


def test_interaction_id_in_provider_specific_fields(lyria_config):
    result = _call_transform(lyria_config, SAMPLE_PAYLOAD)
    psf = result.choices[0].message.provider_specific_fields
    assert psf["interaction_id"] == "uD8sarr0MZ6BvPgP_LDMyQI"


def test_audio_mime_type_in_provider_specific_fields(lyria_config):
    result = _call_transform(lyria_config, SAMPLE_PAYLOAD)
    psf = result.choices[0].message.provider_specific_fields
    assert psf["audio_mime_type"] == "audio/mpeg"


def test_message_content_is_none(lyria_config):
    """OpenAI spec: content=None when audio is returned."""
    result = _call_transform(lyria_config, SAMPLE_PAYLOAD)
    assert result.choices[0].message.content is None


def test_finish_reason_is_stop(lyria_config):
    result = _call_transform(lyria_config, SAMPLE_PAYLOAD)
    assert result.choices[0].finish_reason == "stop"


def test_cost_in_hidden_params(lyria_config):
    with patch("litellm.get_model_info") as mock_get_info:
        mock_get_info.return_value = {"output_cost_per_image": 0.08}
        result = _call_transform(lyria_config, SAMPLE_PAYLOAD)

    assert result._hidden_params is not None
    assert result._hidden_params.get("response_cost") == 0.08
    mock_get_info.assert_called_once_with(
        model="vertex_ai/lyria-3-pro-preview", custom_llm_provider="vertex_ai"
    )


def test_cost_falls_back_to_zero_on_model_info_error(lyria_config):
    with patch("litellm.get_model_info", side_effect=Exception("no model")):
        result = _call_transform(lyria_config, SAMPLE_PAYLOAD)

    assert result._hidden_params.get("response_cost") == 0.0


def test_clip_cost_injects_004_for_clip_model(lyria_config):
    """The clip model id resolves its output_cost_per_image into response_cost."""
    with patch("litellm.get_model_info") as mock_get_info:
        mock_get_info.return_value = {"output_cost_per_image": 0.04}
        result = lyria_config.transform_response(
            model="vertex_ai/lyria-3-clip-preview",
            raw_response=_make_raw_response(SAMPLE_PAYLOAD),
            model_response=_make_model_response(),
            logging_obj=MagicMock(),
            request_data={},
            messages=[{"role": "user", "content": "make music"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
    assert result._hidden_params.get("response_cost") == 0.04
    mock_get_info.assert_called_once_with(
        model="vertex_ai/lyria-3-clip-preview", custom_llm_provider="vertex_ai"
    )


def test_caption_absent_not_in_provider_specific_fields(lyria_config):
    payload = {
        **SAMPLE_PAYLOAD,
        "outputs": [
            {"type": "text", "text": "only lyrics"},
            {"type": "audio", "mime_type": "audio/mpeg", "data": SAMPLE_AUDIO_B64},
        ],
    }
    result = _call_transform(lyria_config, payload)
    psf = result.choices[0].message.provider_specific_fields
    assert "caption" not in psf
    assert result.choices[0].message.audio.transcript == "only lyrics"


def test_interaction_id_absent_not_in_provider_specific_fields(lyria_config):
    payload = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "id"}
    result = _call_transform(lyria_config, payload)
    psf = result.choices[0].message.provider_specific_fields
    assert "interaction_id" not in psf


def test_non_default_mime_type_preserved(lyria_config):
    payload = {
        **SAMPLE_PAYLOAD,
        "outputs": [
            {"type": "text", "text": "lyrics"},
            {"type": "audio", "mime_type": "audio/wav", "data": SAMPLE_AUDIO_B64},
        ],
    }
    result = _call_transform(lyria_config, payload)
    psf = result.choices[0].message.provider_specific_fields
    assert psf["audio_mime_type"] == "audio/wav"


def test_non_dict_output_items_are_skipped(lyria_config):
    payload = {
        **SAMPLE_PAYLOAD,
        "outputs": [
            "garbage-string",
            None,
            {"type": "text", "text": "real lyrics"},
            {"type": "audio", "mime_type": "audio/mpeg", "data": SAMPLE_AUDIO_B64},
        ],
    }
    result = _call_transform(lyria_config, payload)
    assert result.choices[0].message.audio.data == SAMPLE_AUDIO_B64
    assert result.choices[0].message.audio.transcript == "real lyrics"


def test_raises_on_empty_outputs(lyria_config):
    payload = {**SAMPLE_PAYLOAD, "outputs": []}
    with pytest.raises(LyriaError) as exc_info:
        _call_transform(lyria_config, payload)
    assert exc_info.value.status_code == 500
    assert "no outputs" in exc_info.value.message


def test_raises_on_missing_outputs_key(lyria_config):
    payload = {"id": "x", "status": "completed"}
    with pytest.raises(LyriaError) as exc_info:
        _call_transform(lyria_config, payload)
    assert exc_info.value.status_code == 500


def test_raises_when_no_audio_output(lyria_config):
    payload = {
        **SAMPLE_PAYLOAD,
        "outputs": [
            {"type": "text", "text": "lyrics"},
        ],
    }
    with pytest.raises(LyriaError) as exc_info:
        _call_transform(lyria_config, payload)
    assert "no audio output" in exc_info.value.message


def test_raises_on_invalid_json(lyria_config):
    mock_resp = MagicMock()
    mock_resp.json.side_effect = ValueError("bad json")
    mock_resp.text = "not json"

    with pytest.raises(LyriaError) as exc_info:
        lyria_config.transform_response(
            model="vertex_ai/lyria-3-pro-preview",
            raw_response=mock_resp,
            model_response=_make_model_response(),
            logging_obj=MagicMock(),
            request_data={},
            messages=[{"role": "user", "content": "make music"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
    assert exc_info.value.status_code == 500
    assert "Failed to parse" in exc_info.value.message


def test_expires_at_is_24h_from_now(lyria_config):
    before = int(time.time())
    result = _call_transform(lyria_config, SAMPLE_PAYLOAD)
    after = int(time.time())
    audio = result.choices[0].message.audio
    # expires_at should be ~24h (86400s) from now
    assert before + 86400 <= audio.expires_at <= after + 86400 + 5


def test_transform_request_strips_provider_prefix(lyria_config):
    payload = lyria_config.transform_request(
        model="vertex_ai/lyria-3-pro-preview",
        messages=[{"role": "user", "content": "upbeat rock song"}],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert payload["model"] == "lyria-3-pro-preview"
    assert payload["input"] == [{"type": "text", "text": "upbeat rock song"}]


def test_transform_request_ignores_system_message(lyria_config):
    """System message must not be used as the music prompt."""
    payload = lyria_config.transform_request(
        model="vertex_ai/lyria-3-pro-preview",
        messages=[
            {"role": "system", "content": "you are a music assistant"},
            {"role": "user", "content": "upbeat rock song"},
        ],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert payload["input"] == [{"type": "text", "text": "upbeat rock song"}]


def test_transform_request_raises_when_only_system_message(lyria_config):
    with pytest.raises(LyriaError) as exc_info:
        lyria_config.transform_request(
            model="vertex_ai/lyria-3-pro-preview",
            messages=[{"role": "system", "content": "you are a music assistant"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
    assert exc_info.value.status_code == 400


def test_transform_request_uses_last_non_system_message(lyria_config):
    """When multiple non-system messages are passed, the last one is the prompt."""
    payload = lyria_config.transform_request(
        model="vertex_ai/lyria-3-pro-preview",
        messages=[
            {"role": "user", "content": "first idea"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "final prompt"},
        ],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert payload["input"] == [{"type": "text", "text": "final prompt"}]


def test_transform_request_raises_on_empty_prompt(lyria_config):
    with pytest.raises(LyriaError) as exc_info:
        lyria_config.transform_request(
            model="vertex_ai/lyria-3-pro-preview",
            messages=[{"role": "user", "content": ""}],
            optional_params={},
            litellm_params={},
            headers={},
        )
    assert exc_info.value.status_code == 400
    assert "empty" in exc_info.value.message.lower()


def test_map_openai_params_passes_extra_body_through(lyria_config):
    """Lyria does not strip extra_body; the shared handler owns that merge."""
    extra = {"model": "override", "input": "x"}
    out = lyria_config.map_openai_params(
        non_default_params={},
        optional_params={"extra_body": extra},
        model="vertex_ai/lyria-3-pro-preview",
        drop_params=False,
    )
    assert out["extra_body"] == {"model": "override", "input": "x"}


def test_get_complete_url_raises_without_project(lyria_config):
    with pytest.raises(LyriaError) as exc_info:
        lyria_config.get_complete_url(
            api_base=None,
            api_key=None,
            model="lyria-3-pro-preview",
            optional_params={},
            litellm_params={},
        )
    assert exc_info.value.status_code == 400
    assert "vertex_project" in exc_info.value.message


def test_get_complete_url_includes_project(lyria_config):
    with patch.object(
        lyria_config, "safe_get_vertex_ai_project", return_value="my-project"
    ):
        url = lyria_config.get_complete_url(
            api_base=None,
            api_key=None,
            model="lyria-3-pro-preview",
            optional_params={},
            litellm_params={},
        )
    assert "my-project" in url
    assert "interactions" in url
    assert "v1beta1" in url


def test_supports_stream_param_in_request_body_is_false(lyria_config):
    """Lyria API does not accept a stream field in the request body."""
    assert lyria_config.supports_stream_param_in_request_body is False


def test_should_fake_stream_when_stream_true(lyria_config):
    """Lyria must fake-stream: it returns a single JSON object, not SSE."""
    assert (
        lyria_config.should_fake_stream(
            model="vertex_ai/lyria-3-pro-preview",
            stream=True,
            custom_llm_provider="vertex_ai",
        )
        is True
    )


def test_should_not_fake_stream_when_stream_false(lyria_config):
    """Non-streaming calls should not trigger fake-stream."""
    assert (
        lyria_config.should_fake_stream(
            model="vertex_ai/lyria-3-pro-preview",
            stream=False,
            custom_llm_provider="vertex_ai",
        )
        is False
    )


def test_should_not_fake_stream_when_stream_none(lyria_config):
    """Default (no stream param) should not trigger fake-stream."""
    assert (
        lyria_config.should_fake_stream(
            model="vertex_ai/lyria-3-pro-preview",
            stream=None,
            custom_llm_provider="vertex_ai",
        )
        is False
    )
