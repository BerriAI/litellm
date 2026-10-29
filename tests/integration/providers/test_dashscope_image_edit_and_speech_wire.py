import base64
import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, list_value, object_value
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._dashscope_video_support import deployed, json_reply, rate, settled_spend
from pydantic import JsonValue

_API_KEY: Final = "synthetic-dashscope-key"
_MULTIMODAL_PATH: Final = "/api/v1/services/aigc/multimodal-generation/generation"
_SYNTHESIZER_PATH: Final = "/api/v1/services/audio/tts/SpeechSynthesizer"
_OSS_PATH: Final = "/oss/"
_PNG: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_PNG_DATA_URI: Final = f"data:image/png;base64,{base64.b64encode(_PNG).decode()}"
_EDITED_URL: Final = "https://dashscope-result.example.com/edited.png"
_PROMPT: Final = "Turn the red circle into a blue square"
_TEXT: Final = "Hello from LiteLLM."
_BILLED_CHARACTERS: Final = len(_TEXT)
_WAV: Final = b"RIFF\x24\x00\x00\x00WAVEfmt " + uuid.uuid4().bytes
_MP3: Final = b"ID3\x04\x00\x00\x00\x00\x00\x00" + uuid.uuid4().bytes
_SIGNED_QUERY: Final = "Expires=1793316671&OSSAccessKeyId=LTAI5t&Signature=ab%2Bc%3D"
_QWEN_TTS: Final = "dashscope-qwen-tts"
_QWEN_AUDIO_TTS: Final = "dashscope-qwen-audio-tts"
_INPUT_TOKENS: Final = 8
_OUTPUT_TOKENS: Final = 30
_OWNED_TIMEOUT: Final = int(2 * graceful_stop_seconds() + 120)


def _edit_api(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", _MULTIMODAL_PATH), (request.method, request.target)
    return json_reply(
        {
            "output": {"choices": [{"message": {"role": "assistant", "content": [{"image": _EDITED_URL}]}}]},
            "usage": {"width": 1024, "height": 1024, "image_count": 1},
        }
    )


def _synthesized(request: Request, file_name: str, usage: dict[str, JsonValue]) -> Reply:
    audio_url: Final = f"http://{request.headers['host']}{_OSS_PATH}{file_name}?{_SIGNED_QUERY}"
    return json_reply({"output": {"audio": {"url": audio_url, "data": "", "id": "audio-1"}}, "usage": usage})


def _speech_api(request: Request) -> Reply:
    if request.method == "GET" and request.target.startswith(f"{_OSS_PATH}speech.wav"):
        return Reply(body=_WAV, content_type="audio/wav")
    if request.method == "GET" and request.target.startswith(f"{_OSS_PATH}speech.mp3"):
        return Reply(body=_MP3, content_type="audio/mpeg")
    if (request.method, request.target) == ("POST", _MULTIMODAL_PATH):
        return _synthesized(request, "speech.wav", {"characters": _BILLED_CHARACTERS})
    assert (request.method, request.target) == ("POST", _SYNTHESIZER_PATH), (request.method, request.target)
    return _synthesized(
        request,
        "speech.mp3",
        {"characters": _BILLED_CHARACTERS, "input_tokens": _INPUT_TOKENS, "output_tokens": _OUTPUT_TOKENS},
    )


def _pooled_database_url(url: str) -> str:
    parts: Final = urlsplit(url)
    query: Final = "&".join(part for part in (parts.query, "connection_limit=5") if part)
    return urlunsplit(parts._replace(query=query))


def _tts_deployment(name: str, model: str, api_base: str) -> dict[str, JsonValue]:
    return {"model_name": name, "litellm_params": {"model": model, "api_base": api_base, "api_key": _API_KEY}}


def _error_message(body: bytes) -> str:
    return str(object_value(JSON_OBJECT.validate_json(body)["error"])["message"])


@pytest.fixture(scope="module")
def rig() -> Iterator[Gateway]:
    with gateway_from_environment() as gateway:
        yield gateway


@pytest.fixture(scope="module")
def speech_wire() -> Iterator[Wire]:
    with wire_server(_speech_api) as served:
        yield served


@pytest.fixture(scope="module")
def speech_proxy(rig: Gateway, speech_wire: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("dashscope-speech")
    config: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    path: Final = directory / "dashscope-speech.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "model_list": [
                    _tts_deployment(_QWEN_TTS, "dashscope/qwen3-tts-flash", speech_wire.url),
                    _tts_deployment(_QWEN_AUDIO_TTS, "dashscope/qwen-audio-3.1-tts-flash", speech_wire.url),
                ],
                "general_settings": {
                    **object_value(config["general_settings"]),
                    "user_url_allowed_hosts": ["127.0.0.1"],
                },
            }
        )
    )
    with owned_proxy(
        rig, directory, {"DATABASE_URL": _pooled_database_url(os.environ["DATABASE_URL"])}, config=path
    ) as owned:
        yield owned


def test_a_multipart_edit_reaches_dashscope_as_inline_images_and_bills_the_edited_image(gateway: Gateway) -> None:
    with wire_server(_edit_api) as wire, gateway.scenario() as scenario:
        model: Final = deployed(
            gateway,
            scenario,
            model="dashscope/qwen-image-2.0",
            api_base=f"{wire.url}{_MULTIMODAL_PATH}",
            api_key=_API_KEY,
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.client.post(
            "/v1/images/edits",
            data={"model": model, "prompt": _PROMPT, "size": "1024x1024", "negative_prompt": "blurry"},
            files=[("image[]", ("a.png", _PNG, "image/png")), ("image[]", ("b.png", _PNG, "image/png"))],
            headers={"Authorization": f"Bearer {key}"},
        )

        assert response.status_code == 200, response.text
        data: Final = list_value(JSON_OBJECT.validate_json(response.content)["data"])
        assert [object_value(item)["url"] for item in data] == [_EDITED_URL]
        (request,) = wire.drain()
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        assert json.loads(request.body) == {
            "model": "qwen-image-2.0",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"image": _PNG_DATA_URI}, {"image": _PNG_DATA_URI}, {"text": _PROMPT}],
                    }
                ]
            },
            "parameters": {"size": "1024*1024", "negative_prompt": "blurry"},
        }
        settled_spend(key, rate("dashscope", "output_cost_per_image", model="qwen-image-2.0"))


def test_a_fourth_image_is_refused_with_a_400_before_dashscope_is_called(gateway: Gateway) -> None:
    with wire_server(_edit_api) as wire, gateway.scenario() as scenario:
        model: Final = deployed(
            gateway,
            scenario,
            model="dashscope/qwen-image-2.0",
            api_base=f"{wire.url}{_MULTIMODAL_PATH}",
            api_key=_API_KEY,
        )
        response: Final = gateway.client.post(
            "/v1/images/edits",
            data={"model": model, "prompt": _PROMPT},
            files=[("image[]", (f"{index}.png", _PNG, "image/png")) for index in range(4)],
            headers={"Authorization": f"Bearer {gateway.key}"},
        )

        assert response.status_code == 400, response.text
        assert "at most 3 input images, got 4" in _error_message(response.content)
        assert wire.drain() == ()


@pytest.mark.timeout(_OWNED_TIMEOUT)
def test_qwen3_tts_returns_the_downloaded_wav_and_bills_each_input_character(
    speech_proxy: Gateway, speech_wire: Wire
) -> None:
    speech_wire.drain()
    with speech_proxy.scenario() as scenario:
        key: Final = scenario.key(models=[_QWEN_TTS])
        response: Final = speech_proxy.request(
            "POST", "/v1/audio/speech", {"model": _QWEN_TTS, "input": _TEXT, "voice": "Cherry"}, key=key
        )

        assert response.status_code == 200, response.text
        assert response.content == _WAV
        assert response.headers["content-type"] == "audio/wav"
        synthesis, download = speech_wire.drain()
        assert (synthesis.method, synthesis.target) == ("POST", _MULTIMODAL_PATH)
        assert synthesis.headers["authorization"] == f"Bearer {_API_KEY}"
        assert json.loads(synthesis.body) == {"model": "qwen3-tts-flash", "input": {"text": _TEXT, "voice": "Cherry"}}
        assert (download.method, download.target) == ("GET", f"{_OSS_PATH}speech.wav?{_SIGNED_QUERY}")
        assert "authorization" not in download.headers
        settled_spend(key, _BILLED_CHARACTERS * rate("dashscope", "input_cost_per_character", "qwen3-tts-flash"))


@pytest.mark.timeout(_OWNED_TIMEOUT)
def test_qwen_audio_tts_sends_format_and_instruction_returns_the_mp3_and_bills_reported_tokens(
    speech_proxy: Gateway, speech_wire: Wire
) -> None:
    speech_wire.drain()
    model: Final = "qwen-audio-3.1-tts-flash"
    with speech_proxy.scenario() as scenario:
        key: Final = scenario.key(models=[_QWEN_AUDIO_TTS])
        response: Final = speech_proxy.request(
            "POST",
            "/v1/audio/speech",
            {
                "model": _QWEN_AUDIO_TTS,
                "input": _TEXT,
                "voice": "longanhuan_v3.1",
                "response_format": "mp3",
                "instructions": "Speak slowly.",
            },
            key=key,
        )

        assert response.status_code == 200, response.text
        assert response.content == _MP3
        assert response.headers["content-type"] == "audio/mpeg"
        synthesis, download = speech_wire.drain()
        assert (synthesis.method, synthesis.target) == ("POST", _SYNTHESIZER_PATH)
        assert json.loads(synthesis.body) == {
            "model": model,
            "input": {"text": _TEXT, "voice": "longanhuan_v3.1", "format": "mp3", "instruction": "Speak slowly."},
        }
        assert (download.method, download.target) == ("GET", f"{_OSS_PATH}speech.mp3?{_SIGNED_QUERY}")
        settled_spend(
            key,
            _INPUT_TOKENS * rate("dashscope", "input_cost_per_token", model)
            + _OUTPUT_TOKENS * rate("dashscope", "output_cost_per_token", model),
        )


@pytest.mark.timeout(_OWNED_TIMEOUT)
def test_a_format_qwen3_tts_cannot_produce_answers_400_without_reaching_dashscope(
    speech_proxy: Gateway, speech_wire: Wire
) -> None:
    speech_wire.drain()
    refused: Final = speech_proxy.request(
        "POST", "/v1/audio/speech", {"model": _QWEN_TTS, "input": _TEXT, "voice": "Cherry", "response_format": "mp3"}
    )
    served: Final = speech_proxy.request(
        "POST", "/v1/audio/speech", {"model": _QWEN_TTS, "input": _TEXT, "voice": "Cherry", "response_format": "wav"}
    )

    assert refused.status_code == 400, refused.text
    assert "response_format='mp3'" in _error_message(refused.content)
    assert served.status_code == 200, served.text
    assert [(request.method, request.target.split("?")[0]) for request in speech_wire.drain()] == [
        ("POST", _MULTIMODAL_PATH),
        ("GET", f"{_OSS_PATH}speech.wav"),
    ]
