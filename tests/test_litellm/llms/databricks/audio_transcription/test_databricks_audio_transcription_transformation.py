import base64
import io
import json
import sys
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.databricks.audio_transcription.transformation import (
    DatabricksAudioTranscriptionConfig,
)
from litellm.llms.databricks.common_utils import DatabricksException
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

CONFIG = DatabricksAudioTranscriptionConfig()

WAV_BYTES = b"RIFF" + b"\x00" * 64

API_BASE = "https://adb-123.azuredatabricks.net/serving-endpoints"

BARE_HOST = "https://adb-123.azuredatabricks.net"

INVOKE_URL = f"{API_BASE}/whisper-t/invocations"

TRANSCRIPT = "The quick brown fox jumps over the lazy dog."


@pytest.fixture(autouse=True)
def clean_databricks_env(monkeypatch):
    for var in (
        "DATABRICKS_API_BASE",
        "DATABRICKS_API_KEY",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)


def test_transform_request_builds_invoke_body():
    result = CONFIG.transform_audio_transcription_request(
        model="databricks/whisper-t",
        audio_file=WAV_BYTES,
        optional_params={},
        litellm_params={},
    )

    assert isinstance(result, AudioTranscriptionRequestData)
    assert result.files is None
    assert result.content_type == "application/json"
    assert list(result.data) == ["inputs"]
    assert base64.b64decode(result.data["inputs"][0]) == WAV_BYTES


@pytest.mark.parametrize(
    "audio_file",
    [
        WAV_BYTES,
        io.BytesIO(WAV_BYTES),
        ("sample.wav", WAV_BYTES, "audio/wav"),
    ],
    ids=["bytes", "file-like", "tuple"],
)
def test_transform_request_file_variants(audio_file):
    result = CONFIG.transform_audio_transcription_request(
        model="databricks/whisper-t",
        audio_file=audio_file,
        optional_params={},
        litellm_params={},
    )

    assert base64.b64decode(result.data["inputs"][0]) == WAV_BYTES
    assert result.files is None


@pytest.mark.parametrize(
    "api_base, model, expected",
    [
        (API_BASE, "databricks/whisper-t", INVOKE_URL),
        (f"{API_BASE}/", "databricks/whisper-t", INVOKE_URL),
        (BARE_HOST, "whisper-t", INVOKE_URL),
        (f"{BARE_HOST}/", "whisper-t", INVOKE_URL),
    ],
    ids=[
        "serving-endpoints-base",
        "serving-endpoints-trailing-slash",
        "bare-host-normalized",
        "bare-host-trailing-slash",
    ],
)
def test_get_complete_url(api_base, model, expected):
    url = CONFIG.get_complete_url(
        api_base=api_base,
        api_key=None,
        model=model,
        optional_params={},
        litellm_params={},
    )

    assert url == expected


def test_get_complete_url_from_env(monkeypatch):
    monkeypatch.setenv("DATABRICKS_API_BASE", API_BASE)
    url = CONFIG.get_complete_url(
        api_base=None,
        api_key=None,
        model="databricks/whisper-t",
        optional_params={},
        litellm_params={},
    )

    assert url == INVOKE_URL


def test_get_complete_url_rejects_path_shaping():
    with pytest.raises(DatabricksException, match="Invalid Databricks endpoint name") as exc_info:
        CONFIG.get_complete_url(
            api_base=API_BASE,
            api_key=None,
            model="databricks/../x",
            optional_params={},
            litellm_params={},
        )

    assert exc_info.value.status_code == 400


def test_get_complete_url_no_base_raises(monkeypatch):
    monkeypatch.delenv("DATABRICKS_API_BASE", raising=False)
    with patch.dict(sys.modules, {"databricks": None}):
        with pytest.raises(DatabricksException, match="DATABRICKS_API_BASE") as exc_info:
            CONFIG.get_complete_url(
                api_base=None,
                api_key=None,
                model="databricks/whisper-t",
                optional_params={},
                litellm_params={},
            )

    assert exc_info.value.status_code == 400


def test_validate_environment_sets_bearer_and_json():
    headers = CONFIG.validate_environment(
        headers={},
        model="databricks/whisper-t",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="dapi-test",
        api_base=API_BASE,
    )

    assert headers["Authorization"] == "Bearer dapi-test"
    assert headers["Content-Type"] == "application/json"


def test_validate_environment_resolves_base_from_env(monkeypatch):
    # the audio handler passes api_base=None when only the env var is set
    monkeypatch.setenv("DATABRICKS_API_BASE", API_BASE)
    headers = CONFIG.validate_environment(
        headers={},
        model="databricks/whisper-t",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="dapi-test",
        api_base=None,
    )

    assert headers["Authorization"] == "Bearer dapi-test"
    assert headers["Content-Type"] == "application/json"


def test_validate_environment_m2m(monkeypatch):
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "client-id")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("DATABRICKS_API_BASE", API_BASE)
    with patch.object(CONFIG, "_get_oauth_m2m_token", return_value="m2m-token") as mock_token:
        headers = CONFIG.validate_environment(
            headers={},
            model="databricks/whisper-t",
            messages=[],
            optional_params={},
            litellm_params={},
        )

    mock_token.assert_called_once_with(API_BASE, "client-id", "client-secret")
    assert headers["Authorization"] == "Bearer m2m-token"
    assert headers["Content-Type"] == "application/json"


def test_validate_environment_no_credentials_raises():
    with patch.dict(sys.modules, {"databricks": None}):
        with pytest.raises(DatabricksException, match="databricks-sdk") as exc_info:
            CONFIG.validate_environment(
                headers={},
                model="databricks/whisper-t",
                messages=[],
                optional_params={},
                litellm_params={},
            )

    assert exc_info.value.status_code == 400


def test_transform_response_predictions_list():
    raw = httpx.Response(
        200,
        json={"predictions": [TRANSCRIPT]},
        request=httpx.Request("POST", INVOKE_URL),
    )
    response = CONFIG.transform_audio_transcription_response(raw_response=raw)

    assert response.text == TRANSCRIPT
    assert response["task"] == "transcribe"
    assert response._hidden_params == {"predictions": [TRANSCRIPT]}


@pytest.mark.parametrize(
    "payload, expected_text",
    [
        ({"predictions": TRANSCRIPT}, TRANSCRIPT),
        ({"predictions": [{"text": TRANSCRIPT}]}, TRANSCRIPT),
    ],
    ids=["bare-string-predictions", "dict-element-predictions"],
)
def test_transform_response_predictions_variants(payload, expected_text):
    raw = httpx.Response(200, json=payload, request=httpx.Request("POST", INVOKE_URL))
    response = CONFIG.transform_audio_transcription_response(raw_response=raw)

    assert response.text == expected_text


def test_transform_response_empty_transcript_is_valid():
    # silent audio legitimately transcribes to an empty string
    raw = httpx.Response(200, json={"predictions": [""]}, request=httpx.Request("POST", INVOKE_URL))
    response = CONFIG.transform_audio_transcription_response(raw_response=raw)

    assert response.text == ""


@pytest.mark.parametrize("payload", [{"predictions": []}, {}], ids=["empty-list", "missing-key"])
def test_transform_response_empty_predictions_raises(payload):
    raw = httpx.Response(200, json=payload, request=httpx.Request("POST", INVOKE_URL))
    with pytest.raises(DatabricksException, match="no prediction text") as exc_info:
        CONFIG.transform_audio_transcription_response(raw_response=raw)

    assert exc_info.value.status_code == 500


def test_transform_response_non_json_body_raises():
    raw = httpx.Response(
        200,
        content=b"<html>bad gateway</html>",
        request=httpx.Request("POST", INVOKE_URL),
    )
    with pytest.raises(DatabricksException, match="Error parsing Databricks transcription response") as exc_info:
        CONFIG.transform_audio_transcription_response(raw_response=raw)

    assert exc_info.value.status_code == 500


@pytest.mark.parametrize("status_code", [401, 404])
def test_transform_response_error_status_keeps_code(status_code):
    raw = httpx.Response(
        status_code,
        json={"error_code": "ENDPOINT_NOT_FOUND", "message": "Endpoint not found"},
        request=httpx.Request("POST", INVOKE_URL),
    )
    with pytest.raises(DatabricksException) as exc_info:
        CONFIG.transform_audio_transcription_response(raw_response=raw)

    assert exc_info.value.status_code == status_code
    assert "Endpoint not found" in exc_info.value.message
    # the provider error_code rides in the message (no error_code field on BaseLLMException)
    assert "ENDPOINT_NOT_FOUND" in exc_info.value.message


def test_supported_params_advertise_whisper_set():
    assert CONFIG.get_supported_openai_params(model="whisper-t") == [
        "language",
        "prompt",
        "response_format",
        "temperature",
        "timestamp_granularities",
    ]
    mapped = CONFIG.map_openai_params(
        non_default_params={"response_format": "json"},
        optional_params={},
        model="whisper-t",
        drop_params=False,
    )

    assert mapped == {}


def test_transcription_dispatches_to_databricks_invoke(monkeypatch):
    monkeypatch.setenv("DATABRICKS_API_BASE", API_BASE)
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(
            200,
            json={"predictions": ["transcribed text"]},
            request=request,
        )

    http_handler = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler)))
    response = litellm.transcription(
        model="databricks/whisper-t",
        file=("sample.wav", WAV_BYTES, "audio/wav"),
        api_key="dapi-test",
        api_base=API_BASE,
        language="en",
        response_format="json",
        client=http_handler,
    )

    request = captured["request"]
    assert str(request.url) == INVOKE_URL
    assert request.headers["Authorization"] == "Bearer dapi-test"
    assert request.headers["Content-Type"] == "application/json"
    body = json.loads(request.content)
    assert base64.b64decode(body["inputs"][0]) == WAV_BYTES
    assert response.text == "transcribed text"


async def test_atranscription_dispatches_to_databricks_invoke(monkeypatch):
    monkeypatch.setenv("DATABRICKS_API_BASE", API_BASE)
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(
            200,
            json={"predictions": ["transcribed text"]},
            request=request,
        )

    http_handler = AsyncHTTPHandler(transport=httpx.MockTransport(handler))
    response = await litellm.atranscription(
        model="databricks/whisper-t",
        file=("sample.wav", WAV_BYTES, "audio/wav"),
        api_key="dapi-test",
        api_base=API_BASE,
        language="en",
        response_format="json",
        client=http_handler,
    )

    request = captured["request"]
    assert str(request.url) == INVOKE_URL
    assert request.headers["Authorization"] == "Bearer dapi-test"
    assert request.headers["Content-Type"] == "application/json"
    body = json.loads(request.content)
    assert base64.b64decode(body["inputs"][0]) == WAV_BYTES
    assert response.text == "transcribed text"


def test_provider_config_manager_returns_databricks_config():
    config = ProviderConfigManager.get_provider_audio_transcription_config(
        model="whisper-t",
        provider=LlmProviders.DATABRICKS,
    )

    assert isinstance(config, DatabricksAudioTranscriptionConfig)
