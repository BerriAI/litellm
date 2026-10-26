import io
import json
import wave
from uuid import UUID

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.flowspeech.text_to_speech.transformation import FlowSpeechTextToSpeechConfig

REQUEST_ID = "b698824b-de05-486d-8833-05d5c4522a39"
JOB_ID = "11111111-2222-4333-8444-555555555555"


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00\x00" * 24)
    return output.getvalue()


def success(request_id=REQUEST_ID):
    return {
        "id": JOB_ID,
        "requestId": request_id,
        "status": "success",
        "usage": {"billableCharacters": 5, "reservedCredits": 0, "chargedCredits": 5},
        "result": {
            "url": "https://audio.example.com/recording.wav",
            "mimeType": "audio/wav",
            "sampleRate": 24000,
            "channels": 1,
            "bitsPerSample": 16,
        },
    }


def test_maps_single_voice_and_uuid():
    config = FlowSpeechTextToSpeechConfig()
    voice, params = config.map_openai_params("flowspeech-tts", {"response_format": "wav"}, {"voiceName": "Aoede"})
    request = config.transform_text_to_speech_request("flowspeech-tts", "Hello", voice, params, {}, {})
    assert request["dict_body"] == {"input": {"mode": "single", "text": "Hello", "voiceName": "Aoede"}}
    assert UUID(request["headers"]["Idempotency-Key"]).version == 4
    assert config.get_supported_openai_params("flowspeech-tts") == ["voice", "response_format"]


def test_maps_multi_and_recovery_id():
    config = FlowSpeechTextToSpeechConfig()
    segments = [{"speaker": "Alice", "text": "Hi", "voiceName": "Kore"}]
    voice, params = config.map_openai_params(
        "flowspeech-tts", {}, kwargs={"segments": segments, "request_id": REQUEST_ID}
    )
    request = config.transform_text_to_speech_request("flowspeech-tts", "", voice, params, {}, {})
    assert request["dict_body"] == {"input": {"mode": "multi", "segments": segments}}
    assert request["headers"]["Idempotency-Key"] == REQUEST_ID


@pytest.mark.parametrize(
    "params,message",
    [
        ({"speed": 1}, "FlowSpeech does not support instructions or speed"),
        ({"instructions": "calm"}, "FlowSpeech does not support instructions or speed"),
        ({"response_format": "mp3"}, "FlowSpeech only supports response_format='wav'"),
    ],
)
def test_rejects_unsupported_options_even_with_drop_params(params, message):
    with pytest.raises(ValueError, match=message):
        FlowSpeechTextToSpeechConfig().map_openai_params("flowspeech-tts", params, drop_params=True)


@pytest.mark.parametrize("key", ["source", "format", "speakers", "originalText", "prompt", "stream", "extra_body"])
def test_rejects_legacy_and_unsupported_kwargs(key):
    with pytest.raises(ValueError, match="Unsupported"):
        FlowSpeechTextToSpeechConfig().map_openai_params("flowspeech-tts", {}, kwargs={key: "unused"})


@pytest.mark.parametrize("voice", ["", " ", {}, {"voiceName": ""}, {"voiceName": 7}])
def test_rejects_invalid_voice(voice):
    with pytest.raises(ValueError, match="FlowSpeech voice must be a nonempty voice name"):
        FlowSpeechTextToSpeechConfig().map_openai_params("flowspeech-tts", {}, voice)


@pytest.mark.parametrize(
    "params,text,message",
    [
        ({"segments": []}, "", "FlowSpeech requires 1 to 1000 nonempty speaker turns"),
        (
            {"segments": [{"speaker": "A", "text": "Hi", "voiceName": "Kore", "speed": 1}]},
            "",
            "Extra inputs are not permitted",
        ),
        (
            {"segments": [{"speaker": " ", "text": "Hi", "voiceName": "Kore"}]},
            "",
            "FlowSpeech requires 1 to 1000 nonempty speaker turns",
        ),
        (
            {"segments": [{"speaker": "A", "text": "Hi", "voiceName": "Kore"}]},
            "ambiguous",
            "Use input='' with FlowSpeech segments",
        ),
        ({"request_id": "../bad"}, "Hi", "FlowSpeech requires a UUID"),
        ({}, " ", "FlowSpeech input must contain text"),
    ],
)
def test_invalid_requests(params, text, message):
    with pytest.raises(ValueError, match=message):
        FlowSpeechTextToSpeechConfig().transform_text_to_speech_request("flowspeech-tts", text, None, params, {}, {})


def test_conflicting_idempotency_keys():
    with pytest.raises(ValueError, match="must agree"):
        FlowSpeechTextToSpeechConfig().transform_text_to_speech_request(
            "flowspeech-tts", "Hi", None, {"request_id": REQUEST_ID}, {}, {"idempotency-key": JOB_ID}
        )


def test_rejects_multi_with_voice_or_unknown_model():
    with pytest.raises(ValueError, match="segment"):
        FlowSpeechTextToSpeechConfig().map_openai_params("flowspeech-tts", {}, "Kore", kwargs={"segments": []})
    with pytest.raises(ValueError, match="route"):
        FlowSpeechTextToSpeechConfig().map_openai_params("unsupported", {})


def test_bearer_header_overrides_case_insensitively():
    headers = FlowSpeechTextToSpeechConfig().validate_environment(
        {"authorization": "discard", "content-type": "discard", "X-Correlation-Id": "test"},
        "flowspeech-tts",
        "test-key",
    )
    assert httpx.Headers(headers)["authorization"] == "Bearer test-key"
    assert len(httpx.Headers(headers).get_list("authorization")) == 1
    assert headers["Content-Type"] == "application/json"


@pytest.mark.parametrize("key", ["", " ", "two tokens", "bad\nkey", "bad\x00key"])
def test_rejects_invalid_key(key):
    with pytest.raises(ValueError, match="API key"):
        FlowSpeechTextToSpeechConfig().validate_environment({}, "flowspeech-tts", key)


@pytest.mark.parametrize("name", ["Cookie", "Proxy-Authorization", "Host", "content-length"])
def test_rejects_unsafe_api_headers(name):
    with pytest.raises(ValueError, match="does not accept cookie, proxy credential, host or content-length overrides"):
        FlowSpeechTextToSpeechConfig().validate_environment({name: "test"}, "flowspeech-tts", "test-key")


def test_missing_key(monkeypatch):
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.delenv("FLOWSPEECH_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        FlowSpeechTextToSpeechConfig().validate_environment({}, "flowspeech-tts")


@pytest.mark.parametrize("base", ["https://flowspeech.io", "https://flowspeech.io/api/v1/tts/jobs/"])
def test_complete_url(base):
    assert (
        FlowSpeechTextToSpeechConfig().get_complete_url("flowspeech-tts", base, {})
        == "https://flowspeech.io/api/v1/tts/jobs"
    )


@pytest.mark.parametrize(
    "base,message",
    [
        ("http://example.com", "FlowSpeech requires an HTTPS URL without credentials or fragment"),
        ("https://u:p@example.com", "FlowSpeech requires an HTTPS URL without credentials or fragment"),
        ("file:///tmp/test", "FlowSpeech requires an HTTPS URL without credentials or fragment"),
        ("https://example.com/?key=x", "FlowSpeech api_base must be an origin or the /api/v1/tts/jobs endpoint"),
        ("https://example.com/#x", "FlowSpeech requires an HTTPS URL without credentials or fragment"),
        (
            "https://example.com/api/ai/text-to-speech",
            "FlowSpeech api_base must be an origin or the /api/v1/tts/jobs endpoint",
        ),
        ("", "FlowSpeech requires an HTTPS URL without credentials or fragment"),
    ],
)
def test_rejects_invalid_base(base, message):
    with pytest.raises(ValueError, match=message):
        FlowSpeechTextToSpeechConfig().get_complete_url("flowspeech-tts", base, {})


def test_registers_provider():
    from litellm.utils import ProviderConfigManager

    model, provider, _, _ = litellm.get_llm_provider("flowspeech/flowspeech-tts")
    config = ProviderConfigManager.get_provider_text_to_speech_config(model, litellm.LlmProviders.FLOWSPEECH)
    assert model == "flowspeech-tts"
    assert provider == "flowspeech"
    assert isinstance(config, FlowSpeechTextToSpeechConfig)


@pytest.mark.parametrize("multi", [False, True])
def test_speech_dispatch(multi):
    requests = []
    logs = []
    segments = [{"speaker": "A", "text": "Hi", "voiceName": "Kore"}]

    def log_event(details):
        if details.get("log_event_type") == "post_api_call":
            logs.append((details["api_key"], details["original_response"], dict(details["additional_args"])))

    def respond(request):
        requests.append(request)
        if request.method == "POST":
            assert request.url.path == "/api/v1/tts/jobs"
            assert request.headers["authorization"] == "Bearer test-key"
            assert request.headers["idempotency-key"] == REQUEST_ID
            assert json.loads(request.content) == {
                "input": (
                    {"mode": "multi", "segments": segments}
                    if multi
                    else {"mode": "single", "text": "Hello", "voiceName": "Aoede"}
                )
            }
            return httpx.Response(202, json=success())
        assert request.url.host == "audio.example.com"
        assert logs == []
        assert "authorization" not in request.headers and "cookie" not in request.headers
        return httpx.Response(200, content=wav_bytes())

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        result = litellm.speech(
            model="flowspeech/flowspeech-tts",
            input="" if multi else "Hello",
            api_key="test-key",
            response_format="wav",
            request_id=REQUEST_ID,
            timeout=60,
            client=HTTPHandler(client=http_client),
            logger_fn=log_event,
            **({"segments": segments} if multi else {"voice": "Aoede"}),
        )
    assert result.content == wav_bytes()
    assert result.response.headers["content-type"] == "audio/wav"
    assert result.response.headers["x-flowspeech-job-id"] == JOB_ID
    assert len(requests) == 2
    assert len(logs) == 1 and logs[0][0] == ""
    assert json.loads(logs[0][1]) == {key: success()[key] for key in ("id", "requestId", "status", "usage")}
    assert logs[0][2] == {"request_id": REQUEST_ID, "job_id": JOB_ID}
    assert "test-key" not in repr(logs) and "audio.example.com" not in repr(logs)


@pytest.mark.asyncio
async def test_aspeech_dispatch():
    requests = []
    logs = []

    def log_event(details):
        if details.get("log_event_type") == "post_api_call":
            logs.append((details["api_key"], details["original_response"], dict(details["additional_args"])))

    async def respond(request):
        requests.append(request)
        if request.method == "POST":
            assert json.loads(request.content)["input"]["voiceName"] == "Kore"
            return httpx.Response(200, json=success(request.headers["idempotency-key"]))
        assert "authorization" not in request.headers
        assert logs == []
        return httpx.Response(200, content=wav_bytes())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await litellm.aspeech(
            model="flowspeech/flowspeech-tts",
            input="Hello",
            api_key="test-key",
            client=client,
            timeout=60,
            logger_fn=log_event,
        )
    assert result.content == wav_bytes()
    assert len(requests) == 2
    assert len(logs) == 1 and logs[0][0] == ""
    assert json.loads(logs[0][1])["requestId"] == requests[0].headers["idempotency-key"]
    assert logs[0][2]["job_id"] == JOB_ID
    assert "test-key" not in repr(logs) and "audio.example.com" not in repr(logs)
