import io
import json
import logging
import traceback
import wave
from typing import Final
from uuid import UUID

import httpx
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.flowspeech.text_to_speech.transformation import FlowSpeechException, FlowSpeechTextToSpeechConfig

REQUEST_ID = "b698824b-de05-486d-8833-05d5c4522a39"
JOB_ID = "11111111-2222-4333-8444-555555555555"
SECRET_SENTINEL = "credential-value-must-not-appear-672911"


def assert_clean_exception_chain(error: BaseException | None, seen: frozenset[int] = frozenset()) -> None:
    if error is None or id(error) in seen:
        return
    assert SECRET_SENTINEL not in repr(error.args)
    assert SECRET_SENTINEL not in repr(vars(error))
    assert SECRET_SENTINEL not in "".join(traceback.format_exception(error))
    request: Final = getattr(error, "request", None)
    if isinstance(request, httpx.Request):
        assert SECRET_SENTINEL not in str(request.url) + repr(request.headers) + repr(request.content)
    assert_clean_exception_chain(error.__cause__, seen | {id(error)})
    assert_clean_exception_chain(error.__context__, seen | {id(error)})


class FailureRecorder(CustomLogger):
    def __init__(self):
        super().__init__()
        self.errors = []

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.errors.append(
            (str(kwargs["exception"]), kwargs["traceback_exception"], kwargs["standard_logging_object"]["error_str"])
        )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.log_failure_event(kwargs, response_obj, start_time, end_time)


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
        ("https://example.com/?key=x", "FlowSpeech api_base must not contain a query"),
        ("https://example.com/p?", "FlowSpeech api_base must not contain a query"),
        ("https://example.com/#x", "FlowSpeech requires an HTTPS URL without credentials or fragment"),
        (
            "https://example.com/api/ai/text-to-speech",
            "FlowSpeech api_base must not use the legacy text-to-speech endpoint",
        ),
        ("", "FlowSpeech requires an HTTPS URL without credentials or fragment"),
    ],
)
def test_rejects_invalid_base(base, message):
    with pytest.raises(ValueError, match=message):
        FlowSpeechTextToSpeechConfig().get_complete_url("flowspeech-tts", base, {})


@pytest.mark.parametrize(
    "base,expected",
    [
        ("https://gateway.example/p", "https://gateway.example/p/api/v1/tts/jobs"),
        ("https://gateway.example/p/", "https://gateway.example/p/api/v1/tts/jobs"),
        ("https://gateway.example/p/api/v1/tts/jobs/", "https://gateway.example/p/api/v1/tts/jobs"),
        ("https://api/v1/", "https://api/v1/api/v1/tts/jobs"),
    ],
)
def test_gateway_base_preserves_prefix_without_duplicating_endpoint(base, expected):
    assert FlowSpeechTextToSpeechConfig().get_complete_url("flowspeech-tts", base, {}) == expected


@pytest.mark.parametrize("prefix", ["/p%2Fq", "/p%2fq", "/p%252Fq", "/p%3Fq", "/p%23q"])
@pytest.mark.parametrize("suffix", ["/", "/api/v1/tts/jobs", "/api/v1/tts/jobs/"])
def test_gateway_prefix_preserves_encoded_path(prefix, suffix):
    url = FlowSpeechTextToSpeechConfig().get_complete_url(
        "flowspeech-tts", f"https://gateway.example{prefix}{suffix}", {}
    )
    assert httpx.URL(url).raw_path == f"{prefix}/api/v1/tts/jobs".encode("ascii")
    assert httpx.URL(url).query == b""
    assert httpx.URL(url).fragment == ""


@pytest.mark.parametrize(
    "headers",
    [SECRET_SENTINEL, {SECRET_SENTINEL.encode(): "invalid"}, {"Authorization": {"value": SECRET_SENTINEL}}],
)
def test_invalid_headers_raise_no_input_bearing_exception_chain(headers):
    with pytest.raises(ValueError, match="FlowSpeech headers must be a mapping") as error:
        FlowSpeechTextToSpeechConfig().validate_environment(headers, "flowspeech-tts", "test-key")
    assert error.value.__context__ is None
    assert SECRET_SENTINEL not in "".join(traceback.format_exception(error.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize(
    "invalid,message",
    [
        ({"api_key": SECRET_SENTINEL.encode()}, "FlowSpeech API key is required"),
        ({"api_key": {"value": SECRET_SENTINEL}}, "FlowSpeech API key is required"),
        ({"headers": {"Authorization": SECRET_SENTINEL.encode()}}, "FlowSpeech headers must be a mapping"),
        ({"headers": {"Authorization": {"value": SECRET_SENTINEL}}}, "FlowSpeech headers must be a mapping"),
        ({"extra_headers": {"Authorization": SECRET_SENTINEL.encode()}}, "FlowSpeech headers must be a mapping"),
        ({"extra_headers": {"Authorization": {"value": SECRET_SENTINEL}}}, "FlowSpeech headers must be a mapping"),
        ({"headers": SECRET_SENTINEL}, "FlowSpeech headers must be a mapping"),
        ({"extra_headers": SECRET_SENTINEL}, "FlowSpeech headers must be a mapping"),
        ({"headers": {SECRET_SENTINEL.encode(): "unused"}}, "FlowSpeech headers must be a mapping"),
        ({"extra_headers": {SECRET_SENTINEL.encode(): "unused"}}, "FlowSpeech headers must be a mapping"),
        ({"headers": {"X-Correlation-Id": SECRET_SENTINEL.encode()}}, "FlowSpeech headers must be a mapping"),
        ({"headers": {"X-Correlation-Id": {"value": SECRET_SENTINEL}}}, "FlowSpeech headers must be a mapping"),
        ({"extra_headers": {"X-Correlation-Id": SECRET_SENTINEL.encode()}}, "FlowSpeech headers must be a mapping"),
        ({"extra_headers": {"X-Correlation-Id": {"value": SECRET_SENTINEL}}}, "FlowSpeech headers must be a mapping"),
    ],
)
async def test_invalid_credentials_never_reach_errors_or_logs(
    asynchronous,
    positional,
    invalid,
    message,
    monkeypatch,
    caplog,
    capsys,
):
    caplog.set_level(logging.DEBUG, logger="LiteLLM")
    recorder = FailureRecorder()
    monkeypatch.setattr(litellm, "failure_callback", [recorder])
    monkeypatch.setattr(litellm, "_async_failure_callback", [recorder])
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=success())

    kwargs = {"model": "flowspeech/flowspeech-tts", "input": "Hello", "api_key": "test-key", **invalid}
    args: Final = (kwargs["model"], kwargs["input"], None, kwargs["api_key"]) if positional else ()
    call_kwargs: Final = (
        {key: value for key, value in kwargs.items() if key not in ("model", "input", "api_key")}
        if positional
        else kwargs
    )
    if asynchronous:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(litellm.APIConnectionError, match=message) as error:
                await litellm.aspeech(*args, **call_kwargs, client=client)
    else:
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(ValueError, match=message) as error:
                litellm.speech(*args, **call_kwargs, client=client)
    captured = capsys.readouterr()
    assert requests == []
    assert recorder.errors
    assert SECRET_SENTINEL not in str(error.value)
    assert SECRET_SENTINEL not in "".join(traceback.format_exception(error.value))
    assert SECRET_SENTINEL not in repr(recorder.errors)
    assert SECRET_SENTINEL not in caplog.text + captured.out + captured.err
    assert_clean_exception_chain(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize("successful", [False, True])
@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("header_field", ["headers", "extra_headers"])
@pytest.mark.parametrize("header_name", ["Authorization", SECRET_SENTINEL])
async def test_request_log_redaction_preserves_speech_transport(
    asynchronous: bool,
    positional: bool,
    successful: bool,
    verbose: bool,
    header_field: str,
    header_name: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(litellm, "set_verbose", verbose)
    monkeypatch.setattr("litellm._logging.set_verbose", verbose)
    caplog.set_level(logging.WARNING if verbose else logging.DEBUG, logger="LiteLLM")
    headers: Final = {header_name: SECRET_SENTINEL, "X-Correlation-Id": "safe-correlation"}
    args: Final = ("flowspeech/flowspeech-tts", "Hello", "Kore", SECRET_SENTINEL) if positional else ()
    kwargs: Final = (
        {}
        if positional
        else {
            "model": "flowspeech/flowspeech-tts",
            "input": "Hello",
            "voice": "Kore",
            "api_key": SECRET_SENTINEL,
        }
    )
    calls: Final = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "POST":
            assert request.headers["authorization"] == f"Bearer {SECRET_SENTINEL}"
            assert request.headers["x-correlation-id"] == "safe-correlation"
            assert request.headers["idempotency-key"] == REQUEST_ID
            assert json.loads(request.content) == {"input": {"mode": "single", "text": "Hello", "voiceName": "Kore"}}
            if not successful:
                return httpx.Response(
                    400, json={"error": {"code": "synthetic", "message": "synthetic", "retryable": False}}
                )
            return httpx.Response(200, json=success())
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        assert "x-correlation-id" not in request.headers
        return httpx.Response(200, content=wav_bytes())

    async def call():
        if asynchronous:
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                return await litellm.aspeech(
                    *args,
                    **kwargs,
                    **{header_field: headers},
                    request_id=REQUEST_ID,
                    max_retries=0,
                    client=client,
                )
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            return litellm.speech(
                *args,
                **kwargs,
                **{header_field: headers},
                request_id=REQUEST_ID,
                max_retries=0,
                client=client,
            )

    if successful:
        result: Final = await call()
        assert result.content == wav_bytes()
    else:
        with pytest.raises(litellm.BadRequestError if asynchronous else FlowSpeechException) as error:
            await call()
        assert_clean_exception_chain(error.value)
        assert error.value.status_code == 400
    captured: Final = capsys.readouterr()
    output: Final = caplog.text + captured.out + captured.err
    assert SECRET_SENTINEL not in output
    assert "REDACTED" in output
    assert ("safe-correlation" in output) is (header_name == "Authorization")
    assert "Hello" in output and "Kore" in output
    assert calls == (["POST", "GET"] if successful else ["POST"])
    assert headers == {header_name: SECRET_SENTINEL, "X-Correlation-Id": "safe-correlation"}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("debug", [False, True])
async def test_positional_key_debug_controls(
    asynchronous: bool,
    debug: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(litellm, "set_verbose", False)
    monkeypatch.setattr("litellm._logging.set_verbose", False)
    caplog.set_level(logging.DEBUG if debug else logging.ERROR, logger="LiteLLM")
    calls: Final = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        assert request.headers["authorization"] == f"Bearer {SECRET_SENTINEL}"
        return httpx.Response(400, json={"error": {"code": "synthetic", "message": "synthetic", "retryable": False}})

    async def call() -> None:
        if asynchronous:
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                await litellm.aspeech(
                    "flowspeech/flowspeech-tts", "Hello", "Kore", SECRET_SENTINEL, client=client, max_retries=0
                )
        else:
            with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                litellm.speech(
                    "flowspeech/flowspeech-tts", "Hello", "Kore", SECRET_SENTINEL, client=client, max_retries=0
                )

    with pytest.raises(litellm.BadRequestError if asynchronous else FlowSpeechException) as error:
        await call()
    captured: Final = capsys.readouterr()
    assert SECRET_SENTINEL not in caplog.text + captured.out + captured.err
    assert ("Request to litellm:" in caplog.text) is debug
    assert calls == ["POST"]
    assert_clean_exception_chain(error.value)


def test_registers_provider():
    from litellm.utils import ProviderConfigManager

    model, provider, _, _ = litellm.get_llm_provider("flowspeech/flowspeech-tts")
    config = ProviderConfigManager.get_provider_text_to_speech_config(model, litellm.LlmProviders.FLOWSPEECH)
    assert model == "flowspeech-tts"
    assert provider == "flowspeech"
    assert isinstance(config, FlowSpeechTextToSpeechConfig)
    assert "flowspeech/flowspeech-tts" in litellm.model_list
    assert litellm.models_by_provider[provider] == litellm.flowspeech_models
    assert "flowspeech/flowspeech-tts" in litellm.models_by_provider[provider]


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("timeout_args", [{}, {"timeout": None}, {"timeout": 60}])
def test_speech_dispatch(multi, timeout_args):
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
            client=HTTPHandler(client=http_client),
            logger_fn=log_event,
            **timeout_args,
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
@pytest.mark.parametrize("timeout_args", [{}, {"timeout": None}, {"timeout": 60}])
async def test_aspeech_dispatch(timeout_args):
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
            logger_fn=log_event,
            **timeout_args,
        )
    assert result.content == wav_bytes()
    assert len(requests) == 2
    assert len(logs) == 1 and logs[0][0] == ""
    assert json.loads(logs[0][1])["requestId"] == requests[0].headers["idempotency-key"]
    assert logs[0][2]["job_id"] == JOB_ID
    assert "test-key" not in repr(logs) and "audio.example.com" not in repr(logs)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_dispatch_rejects_mismatched_client_before_sending(asynchronous):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=success())

    client_type = httpx.Client if asynchronous else httpx.AsyncClient
    client = client_type(transport=httpx.MockTransport(respond))
    try:
        kwargs = {"model": "flowspeech/flowspeech-tts", "input": "Hello", "api_key": "test-key", "client": client}
        message = "client must match the synchronous or asynchronous speech call"
        if asynchronous:
            with pytest.raises(litellm.APIConnectionError, match=message):
                await litellm.aspeech(**kwargs)
        else:
            with pytest.raises(ValueError, match=message):
                litellm.speech(**kwargs)
        assert requests == []
    finally:
        if isinstance(client, httpx.AsyncClient):
            await client.aclose()
        else:
            client.close()
