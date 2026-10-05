import base64
from typing import Final
from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.minimax.text_to_speech.transformation import MinimaxException, MinimaxTextToSpeechConfig
from litellm.types.llms.openai import HttpxBinaryResponseContent

_AUDIO: Final = b"ID3\x04minimax-audio"
_REQUEST: Final = httpx.Request("POST", "https://api.minimax.io/v1/t2a_v2")


def _transform(payload: object, status_code: int = 200) -> HttpxBinaryResponseContent:
    return MinimaxTextToSpeechConfig().transform_text_to_speech_response(
        model="speech-02-hd",
        raw_response=httpx.Response(
            status_code,
            json=payload,
            request=_REQUEST,
            headers={"content-encoding": "identity", "x-trace": "abc"},
        ),
        logging_obj=Mock(),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"audio": _AUDIO.hex()}, "status": 0, "extra_info": {"audio_length": 5}},
        {"data": {"audio": _AUDIO.hex(), "audio_url": ""}},
        {"data": {"audio": ""}, "audio_file": _AUDIO.hex()},
        {"data": {"audio": None}, "audio_file": base64.b64encode(_AUDIO).decode()},
        {"base_resp": {"status_code": 0}, "audio_file": base64.b64encode(_AUDIO).decode()},
    ],
)
def test_transform_text_to_speech_response_decodes_hex_or_base64_audio(payload: dict[str, object]):
    response = _transform(payload).response

    assert response.status_code == 200
    assert response.content == _AUDIO
    assert response.headers["content-length"] == str(len(_AUDIO))
    assert response.headers["x-trace"] == "abc"
    assert "content-encoding" not in response.headers


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"status": 2, "ced": "invalid api key"}, "invalid api key"),
        ({"status": 2}, "Unknown error"),
        ({"status": 1004, "ced": ""}, "API returned status 1004"),
        ({"status": "failed", "ced": None, "data": {"audio": _AUDIO.hex()}}, "API returned status failed"),
    ],
)
def test_transform_text_to_speech_response_reports_api_status_errors(payload: dict[str, object], detail: str):
    with pytest.raises(MinimaxException) as exc_info:
        _transform(payload, status_code=401)

    assert exc_info.value.message == f"MiniMax TTS error: {detail}"
    assert exc_info.value.status_code == 401


def test_transform_text_to_speech_response_refuses_url_output():
    with pytest.raises(MinimaxException) as exc_info:
        _transform({"data": {"audio_url": "https://cdn.example/a.mp3", "audio": _AUDIO.hex()}})

    assert exc_info.value.message == (
        "URL output format is not yet supported. Use 'hex' format or fetch from URL: https://cdn.example/a.mp3"
    )
    assert exc_info.value.status_code == 500


@pytest.mark.parametrize(
    ("payload", "keys"),
    [
        ({}, []),
        ({"data": {}, "status": 0}, ["data", "status"]),
        ({"data": {"audio": ""}, "audio_file": ""}, ["data", "audio_file"]),
        ({"data": {"audio": None}, "audio_file": None}, ["data", "audio_file"]),
        ({"audio_file": 0, "data": {"audio": []}}, ["audio_file", "data"]),
    ],
)
def test_transform_text_to_speech_response_without_audio_lists_the_response_keys(
    payload: dict[str, object], keys: list[str]
):
    with pytest.raises(MinimaxException) as exc_info:
        _transform(payload)

    assert exc_info.value.message == f"No audio data in MiniMax response. Response keys: {keys}"
    assert exc_info.value.status_code == 500


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        "not an object",
        {"data": None},
        {"data": "not an object"},
        {"data": ["not", "an", "object"]},
        {"data": {"audio": 7}},
        {"data": {"audio": ["49", "44"]}},
        {"audio_file": {"hex": "4944"}},
    ],
)
def test_transform_text_to_speech_response_wraps_malformed_payloads_without_echoing_them(payload: object):
    with pytest.raises(MinimaxException) as exc_info:
        _transform(payload)

    assert exc_info.value.message.startswith("Error processing MiniMax response: ")
    assert "input_value" not in exc_info.value.message
    assert exc_info.value.status_code == 500


def test_transform_text_to_speech_response_reports_undecodable_audio():
    with pytest.raises(MinimaxException) as exc_info:
        _transform({"data": {"audio": "zzz"}})

    assert exc_info.value.message.startswith("Failed to decode audio data: ")
    assert exc_info.value.status_code == 500
