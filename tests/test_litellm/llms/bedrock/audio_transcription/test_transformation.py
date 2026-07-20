import base64
import json

import httpx
import pytest

import litellm
from litellm.llms.bedrock.audio_transcription.transformation import (
    BedrockAudioTranscriptionConfig,
)
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.utils import get_optional_params_transcription


def test_transform_audio_transcription_request_builds_converse_audio_body() -> None:
    config = BedrockAudioTranscriptionConfig()
    audio_bytes = b"audio-bytes"

    request = config.transform_audio_transcription_request(
        model="bedrock/mistral.voxtral-mini-3b-2507",
        audio_file=("recording.mp3", audio_bytes, "audio/mpeg"),
        optional_params={
            "language": "en",
            "prompt": "Speaker names are important",
            "temperature": 0.2,
        },
        litellm_params={},
    )

    assert isinstance(request.data, dict)
    message = request.data["messages"][0]
    audio = message["content"][0]["audio"]
    assert audio["format"] == "mp3"
    assert base64.b64decode(audio["source"]["bytes"]) == audio_bytes
    assert "The audio language is en." in message["content"][1]["text"]
    assert "Speaker names are important" in message["content"][1]["text"]
    assert request.data["system"] == [{"text": "You are a transcription assistant."}]
    assert request.data["inferenceConfig"] == {"maxTokens": 4096, "temperature": 0.2}


def test_transform_audio_transcription_response_concatenates_text_and_hides_raw_response() -> None:
    raw_response = {
        "output": {
            "message": {
                "content": [{"text": "hello "}, {"text": "world"}],
            }
        },
        "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
        "stopReason": "end_turn",
    }
    response = BedrockAudioTranscriptionConfig().transform_audio_transcription_response(
        httpx.Response(200, json=raw_response)
    )

    assert response.text == "hello world"
    assert response._hidden_params == raw_response
    assert response.usage is None


def test_timestamp_granularities_respects_drop_params() -> None:
    with pytest.raises(litellm.exceptions.UnsupportedParamsError):
        get_optional_params_transcription(
            model="bedrock/mistral.voxtral-mini-3b-2507",
            custom_llm_provider="bedrock",
            timestamp_granularities=["word"],
            drop_params=False,
        )

    optional_params = get_optional_params_transcription(
        model="bedrock/mistral.voxtral-mini-3b-2507",
        custom_llm_provider="bedrock",
        timestamp_granularities=["word"],
        drop_params=True,
    )
    assert "timestamp_granularities" not in optional_params


def test_complete_url_normalizes_regional_bedrock_model() -> None:
    url = BedrockAudioTranscriptionConfig().get_complete_url(
        api_base=None,
        api_key=None,
        model="bedrock/us-east-1/mistral.voxtral-mini-3b-2507",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://bedrock-runtime.us-east-1.amazonaws.com/model/mistral.voxtral-mini-3b-2507/converse"


def test_sigv4_headers_sign_serialized_request_body() -> None:
    credentials_module = pytest.importorskip("botocore.credentials")

    config = BedrockAudioTranscriptionConfig()
    body = json.dumps({"messages": [{"role": "user", "content": [{"text": "hello"}]}]})
    headers = config.get_request_headers(
        credentials=credentials_module.Credentials("test-access-key", "test-secret-key"),
        aws_region_name="us-east-1",
        extra_headers=None,
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com/model/test/converse",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    assert headers.headers["Authorization"].startswith("AWS4-HMAC-SHA256")
    assert "X-Amz-Date" in headers.headers
