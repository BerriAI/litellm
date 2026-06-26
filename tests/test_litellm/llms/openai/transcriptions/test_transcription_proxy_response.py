from litellm.proxy.common_utils.openai_endpoint_utils import (
    get_transcription_proxy_response,
)
from litellm.types.utils import TranscriptionResponse


def test_get_transcription_proxy_response_text_format_returns_plain_text():
    response = TranscriptionResponse(text="Hello world.")

    result = get_transcription_proxy_response(
        response,
        "text",
        headers={"x-litellm-call-id": "test-call-id"},
    )

    assert result.media_type.startswith("text/plain")
    assert result.body == b"Hello world."
    assert result.headers["x-litellm-call-id"] == "test-call-id"


def test_get_transcription_proxy_response_json_format_returns_model():
    response = TranscriptionResponse(text="Hello world.")

    result = get_transcription_proxy_response(response, "json")

    assert isinstance(result, TranscriptionResponse)
    assert result.text == "Hello world."


def test_get_transcription_proxy_response_default_format_returns_model():
    response = TranscriptionResponse(text="Hello world.")

    result = get_transcription_proxy_response(response, None)

    assert isinstance(result, TranscriptionResponse)
    assert result.text == "Hello world."
