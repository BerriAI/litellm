import asyncio
import io
import json
import struct
import wave
from datetime import datetime, timezone
from email.utils import format_datetime

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.llms.flowspeech.text_to_speech.handler import FlowSpeechTTS
from litellm.llms.flowspeech.text_to_speech.transformation import FlowSpeechException, FlowSpeechTextToSpeechConfig
from litellm.types.utils import LlmProviders

REQUEST_ID = "b698824b-de05-486d-8833-05d5c4522a39"
JOB_ID = "11111111-2222-4333-8444-555555555555"
URL = "https://flowspeech.io/api/v1/tts/jobs"
AUDIO_URL = "https://audio.example.com/recording.wav"


def wav_bytes(pcm=b"\x00\x00" * 24, metadata=False):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(pcm)
    content = output.getvalue()
    if metadata:
        content = content[:12] + b"JUNK" + struct.pack("<I", 3) + b"abc\x00" + content[12:]
        content = resized_riff(content)
    return content


def resized_riff(content):
    return content[:4] + struct.pack("<I", len(content) - 8) + content[8:]


def changed_wav(offset, replacement):
    content = wav_bytes()
    return content[:offset] + replacement + content[offset + len(replacement) :]


def job(status="success", **overrides):
    return {
        "id": JOB_ID,
        "requestId": REQUEST_ID,
        "status": status,
        "usage": {
            "billableCharacters": 2,
            "reservedCredits": 2 if status in ("queued", "running", "merging", "settling") else 0,
            "chargedCredits": 2 if status == "success" else 0,
        },
        "result": {
            "url": AUDIO_URL,
            "mimeType": "audio/wav",
            "sampleRate": 24000,
            "channels": 1,
            "bitsPerSample": 16,
        }
        if status == "success"
        else None,
        **overrides,
    }


class Clock:
    def __init__(self):
        self.now = 0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay

    async def asleep(self, delay):
        self.sleep(delay)
        await asyncio.sleep(0)


def handler(clock, timeout=120, url=URL, **kwargs):
    return FlowSpeechTTS(
        url,
        {
            "Authorization": "Bearer test-key",
            "Idempotency-Key": REQUEST_ID,
            "Content-Type": "application/json",
            "X-Private-Api-Header": "test",
        },
        {"input": {"mode": "single", "text": "Hi", "voiceName": "Kore"}},
        timeout,
        clock=clock.time,
        sleep=clock.sleep,
        async_sleep=clock.asleep,
        wall_clock=lambda: 0,
        **kwargs,
    )


@pytest.fixture(params=["sync", "async"])
def invoke(request):
    def run(tts, respond):
        options = {
            "transport": httpx.MockTransport(respond),
            "headers": {"Authorization": "discard-default", "X-Client-Secret": "discard"},
            "cookies": {"session": "discard"},
            "auth": ("discard", "discard"),
            "follow_redirects": True,
        }
        if request.param == "sync":
            with httpx.Client(**options) as client:
                return tts.run(client)

        async def async_run():
            async with httpx.AsyncClient(**options) as client:
                return await tts.arun(client)

        return asyncio.run(async_run())

    return run


def test_all_pending_states_and_uncredentialed_download(invoke):
    clock = Clock()
    requests = []
    states = iter(["queued", "running", "merging", "settling", "success"])

    def respond(request):
        requests.append(request)
        assert "cookie" not in request.headers
        assert "x-client-secret" not in request.headers
        if request.url.host == "audio.example.com":
            assert "authorization" not in request.headers
            assert "x-private-api-header" not in request.headers
            assert "idempotency-key" not in request.headers
            return httpx.Response(200, content=wav_bytes())
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.url == (URL if request.method == "POST" else f"{URL}/{JOB_ID}")
        return httpx.Response(
            202 if request.method == "POST" else 200,
            json=job(next(states)),
            headers={"set-cookie": "session=discard-from-server"},
        )

    result = invoke(handler(clock), respond)
    assert result.content == wav_bytes()
    assert result.response.headers["x-flowspeech-request-id"] == REQUEST_ID
    assert "set-cookie" not in result.response.headers
    assert clock.sleeps == [5, 5, 5, 5]
    assert len(requests) == 6
    assert requests[-1].extensions["timeout"]["read"] == 100


@pytest.mark.parametrize("prefix", ["/p%2Fq", "/p%2fq", "/p%252Fq", "/p%3Fq", "/p%23q"])
def test_encoded_gateway_prefix_survives_submission_and_polling(invoke, prefix):
    requests = []
    endpoint = f"{prefix}/api/v1/tts/jobs"
    url = FlowSpeechTextToSpeechConfig().get_complete_url("flowspeech-tts", f"https://gateway.example{prefix}/", {})

    def respond(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(202, json=job("queued"))
        if request.url.host == "gateway.example":
            return httpx.Response(200, json=job())
        return httpx.Response(200, content=wav_bytes())

    assert invoke(handler(Clock(), url=url), respond).content == wav_bytes()
    assert [(request.method, request.url.raw_path) for request in requests] == [
        ("POST", endpoint.encode()),
        ("GET", f"{endpoint}/{JOB_ID}".encode()),
        ("GET", b"/recording.wav"),
    ]
    assert requests[0].headers["Idempotency-Key"] == REQUEST_ID


@pytest.mark.parametrize("failure", ["network", "server", "rate_limit"])
def test_create_retry_preserves_uuid_and_body(invoke, failure):
    clock = Clock()
    requests = []

    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            if failure == "network":
                raise httpx.ReadTimeout("lost response")
            return httpx.Response(
                429 if failure == "rate_limit" else 503,
                headers={"Retry-After": "11"},
                json={"error": {"code": "busy", "message": "wait", "retryable": True}, "traceId": "trace"},
            )
        return httpx.Response(200, json=job()) if request.method == "POST" else httpx.Response(200, content=wav_bytes())

    assert invoke(handler(clock), respond).content == wav_bytes()
    assert requests[0].content == requests[1].content
    assert requests[0].headers["idempotency-key"] == requests[1].headers["idempotency-key"] == REQUEST_ID
    assert clock.sleeps == [5 if failure == "network" else 11]


def test_http_date_retry_after_and_poll_rate_limit(invoke):
    clock = Clock()
    responses = iter(
        [
            httpx.Response(
                202,
                json=job("queued"),
                headers={"Retry-After": format_datetime(datetime.fromtimestamp(12, timezone.utc))},
            ),
            httpx.Response(
                429,
                json={"error": {"code": "rate", "message": "wait", "retryable": True}},
                headers={"Retry-After": "8"},
            ),
            httpx.Response(200, json=job()),
            httpx.Response(200, content=wav_bytes()),
        ]
    )
    invoke(handler(clock), lambda request: next(responses))
    assert clock.sleeps == [12, 8]


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_terminal_failure_is_not_http_success(invoke, status):
    calls = []
    completed = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            200, json=job(status, error={"code": "stopped", "message": "stopped", "retryable": False})
        )

    with pytest.raises(FlowSpeechException, match=f"job {status}.*{REQUEST_ID}.*{JOB_ID}"):
        invoke(handler(Clock(), on_completed=completed.append), respond)
    assert len(calls) == 1
    assert completed == []


@pytest.mark.parametrize(
    "payload,match",
    [
        (job(result=None, resultDeleted=True), "deleted"),
        (job(result=None), "no result"),
        (job(status="mystery"), "invalid job"),
        (job(id="../other"), "invalid job"),
        ({"status": "queued"}, "invalid job"),
        (job(result={"url": AUDIO_URL}), "invalid job"),
        (job(result={**job()["result"], "sampleRate": 48000}), "invalid job"),
        ({"code": 0, "data": {"audioBase64": "test"}}, "invalid job"),
    ],
)
def test_bad_job_responses(invoke, payload, match):
    with pytest.raises(FlowSpeechException, match=match):
        invoke(handler(Clock()), lambda request: httpx.Response(200, json=payload))


def test_rejects_changed_job_id(invoke):
    responses = iter([httpx.Response(202, json=job("queued")), httpx.Response(200, json=job(id=REQUEST_ID))])
    with pytest.raises(FlowSpeechException, match="different job ID"):
        invoke(handler(Clock()), lambda request: next(responses))


@pytest.mark.parametrize("poll", [False, True])
@pytest.mark.parametrize(
    "request_id", [None, 7, "invalid", REQUEST_ID.replace("-", ""), "{" + REQUEST_ID + "}", JOB_ID]
)
def test_rejects_invalid_or_mismatched_request_id_before_audio(invoke, poll, request_id):
    calls, completed = [], []
    responses = iter(([job("queued")] if poll else []) + [job(requestId=request_id)])

    def respond(request):
        calls.append(request)
        assert request.url.host == "flowspeech.io"
        return httpx.Response(200, json=next(responses))

    with pytest.raises(FlowSpeechException):
        invoke(handler(Clock(), on_completed=completed.append), respond)
    assert len(calls) == (2 if poll else 1) and completed == []


@pytest.mark.parametrize("poll", [False, True])
@pytest.mark.parametrize("status", ["queued", "running", "merging", "settling", "success", "failed", "cancelled"])
def test_identity_checked_for_every_job_state(invoke, poll, status):
    responses = iter(([job("queued")] if poll else []) + [job(status, requestId=JOB_ID)])
    with pytest.raises(FlowSpeechException, match="different request ID"):
        invoke(handler(Clock()), lambda request: httpx.Response(200, json=next(responses)))


@pytest.mark.parametrize("poll", [False, True])
@pytest.mark.parametrize("field", ["requestId", "usage"])
def test_job_requires_identity_and_usage(invoke, poll, field):
    payload = {key: value for key, value in job().items() if key != field}
    responses = iter(([job("queued")] if poll else []) + [payload])
    with pytest.raises(FlowSpeechException, match="invalid job"):
        invoke(handler(Clock()), lambda request: httpx.Response(200, json=next(responses)))


@pytest.mark.parametrize("poll", [False, True])
@pytest.mark.parametrize("field", ["billableCharacters", "reservedCredits", "chargedCredits"])
@pytest.mark.parametrize("value", [None, -1, True, "2", 1.5, float("nan"), float("inf")])
def test_rejects_invalid_usage_on_every_response(invoke, poll, field, value):
    payload = job(usage={**job()["usage"], field: value})
    responses = iter(([job("queued")] if poll else []) + [payload])

    def respond(request):
        assert request.url.host == "flowspeech.io"
        return httpx.Response(200, content=json.dumps(next(responses)))

    with pytest.raises(FlowSpeechException, match="invalid job"):
        invoke(handler(Clock()), respond)


@pytest.mark.parametrize("usage", [None, {}, {"billableCharacters": 2, "reservedCredits": 0}])
def test_rejects_missing_usage_counters(invoke, usage):
    with pytest.raises(FlowSpeechException, match="invalid job"):
        invoke(handler(Clock()), lambda request: httpx.Response(200, json=job(usage=usage)))


def test_uppercase_response_ids_are_normalized_and_completion_follows_audio(invoke):
    completed = []
    mixed_case_id = "aa111111-2222-4333-8444-555555555555"
    responses = iter(
        [
            job("queued", requestId=REQUEST_ID.upper(), id=mixed_case_id.upper()),
            job(requestId=REQUEST_ID.upper(), id=mixed_case_id),
        ]
    )

    def respond(request):
        assert completed == []
        if request.url.host == "audio.example.com":
            return httpx.Response(200, content=wav_bytes())
        return httpx.Response(200, json=next(responses))

    result = invoke(handler(Clock(), on_completed=completed.append), respond)
    assert result.content == wav_bytes() and len(completed) == 1
    assert completed[0].requestId == REQUEST_ID and completed[0].id == mixed_case_id


def test_invalid_wav_never_calls_completion(invoke):
    completed = []
    responses = iter([httpx.Response(200, json=job()), httpx.Response(200, content=b"RIFF-invalid")])
    with pytest.raises(FlowSpeechException):
        invoke(handler(Clock(), on_completed=completed.append), lambda request: next(responses))
    assert completed == []


@pytest.mark.parametrize(
    "url",
    [
        "http://audio.example.com/a",
        "https://user:pass@audio.example.com/a",
        "file:///etc/passwd",
        "https://audio.example.com/a#fragment",
    ],
)
def test_rejects_unsafe_download_url(invoke, url):
    with pytest.raises(FlowSpeechException, match="unsafe audio URL"):
        invoke(handler(Clock()), lambda request: httpx.Response(200, json=job(result={**job()["result"], "url": url})))


@pytest.mark.parametrize("download", [False, True])
def test_redirects_rejected_without_following(invoke, download):
    calls = []

    def respond(request):
        calls.append(request)
        if download and len(calls) == 1:
            return httpx.Response(200, json=job())
        return httpx.Response(307, headers={"Location": "https://untrusted.example/collect"})

    with pytest.raises(FlowSpeechException, match="redirect"):
        invoke(handler(Clock()), respond)
    assert len(calls) == (2 if download else 1)


def test_download_retries_do_not_create_another_job(invoke):
    calls = []
    completed = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, json=job())
        if len(calls) == 2:
            raise httpx.ReadError("download interrupted")
        return httpx.Response(200, content=wav_bytes())

    assert invoke(handler(Clock(), on_completed=completed.append), respond).content == wav_bytes()
    assert [request.method for request in calls] == ["POST", "GET", "GET"]
    assert calls[1].url == calls[2].url
    assert len(completed) == 1


def test_download_exhaustion_has_recovery_ids(invoke):
    calls = []

    def respond(request):
        calls.append(request)
        if request.method == "POST":
            return httpx.Response(200, json=job())
        return httpx.Response(503)

    with pytest.raises(FlowSpeechException, match=f"download.*{REQUEST_ID}.*{JOB_ID}"):
        invoke(handler(Clock(), max_retries=1), respond)
    assert [request.method for request in calls] == ["POST", "GET", "GET"]


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 409, 413, 415, 503])
def test_api_errors_preserve_code_trace_and_no_retry(invoke, status):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            status,
            json={
                "error": {"code": "test_error", "message": "test message", "retryable": False},
                "traceId": "trace-id",
            },
        )

    with pytest.raises(FlowSpeechException, match=f"test_error.*trace-id.*{REQUEST_ID}") as exc:
        invoke(handler(Clock()), respond)
    assert exc.value.status_code == status
    assert len(calls) == 1


def test_deadline_before_long_retry_after(invoke):
    clock = Clock()
    with pytest.raises(FlowSpeechException, match="deadline"):
        invoke(
            handler(clock, timeout=10),
            lambda request: httpx.Response(202, json=job("queued"), headers={"Retry-After": "30"}),
        )
    assert clock.sleeps == []


def test_deadline_during_request(invoke):
    clock = Clock()

    def respond(request):
        clock.now = 11
        return httpx.Response(202, json=job("queued"))

    with pytest.raises(FlowSpeechException, match="deadline"):
        invoke(handler(clock, timeout=10), respond)


@pytest.mark.parametrize("stage", ["submission", "download"])
@pytest.mark.parametrize("header_delay,body_delay,expected_reads", [(0.125, 0, 0), (0.075, 0.075, 1)])
def test_expired_stream_is_closed_without_more_work(invoke, stage, header_delay, body_delay, expected_reads):
    clock = Clock()
    completed = []
    requests = []

    class Stream(httpx.SyncByteStream, httpx.AsyncByteStream):
        def __init__(self):
            self.reads = 0
            self.closed = False

        def __iter__(self):
            self.reads += 1
            clock.now += body_delay
            yield wav_bytes() if stage == "download" else json.dumps(job()).encode()
            pytest.fail("must not advance a stream after its budget expires")

        async def __aiter__(self):
            for chunk in self:
                yield chunk

        def close(self):
            self.closed = True

        async def aclose(self):
            self.close()

    stream = Stream()

    def respond(request):
        requests.append(request)
        if stage == "download" and request.method == "POST":
            return httpx.Response(202, json=job())
        clock.now += header_delay
        return httpx.Response(200, stream=stream)

    with pytest.raises(FlowSpeechException, match="total deadline") as error:
        invoke(handler(clock, timeout=0.1, on_completed=completed.append), respond)
    assert error.value.status_code == 408
    assert REQUEST_ID in str(error.value)
    assert f"job_id={JOB_ID if stage == 'download' else 'unknown'}" in str(error.value)
    assert stream.reads == expected_reads
    assert stream.closed
    assert [request.method for request in requests] == (["POST", "GET"] if stage == "download" else ["POST"])
    assert clock.sleeps == []
    assert completed == []


def test_deadline_after_wav_validation_prevents_success_callback(invoke, monkeypatch):
    clock = Clock()
    completed = []
    content = wav_bytes()
    open_wav = wave.open

    def delayed_open(*args, **kwargs):
        clock.now = 11
        return open_wav(*args, **kwargs)

    monkeypatch.setattr(wave, "open", delayed_open)
    responses = iter([httpx.Response(202, json=job()), httpx.Response(200, content=content)])
    with pytest.raises(FlowSpeechException, match="total deadline"):
        invoke(handler(clock, timeout=10, on_completed=completed.append), lambda request: next(responses))
    assert completed == []
    assert clock.sleeps == []


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), httpx.Timeout(None)])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        handler(Clock(), timeout=timeout)


@pytest.mark.parametrize("retries", [-1, 11, 1.5, True])
def test_invalid_retries(retries):
    with pytest.raises(ValueError, match="max_retries"):
        handler(Clock(), max_retries=retries)


@pytest.mark.parametrize(
    "content,headers",
    [
        (b"not wav", {}),
        (wav_bytes(b""), {}),
        (wav_bytes()[:-2], {}),
        (wav_bytes(b"\x00" * 49), {}),
        (changed_wav(4, struct.pack("<I", 9999)), {}),
        (changed_wav(34, struct.pack("<H", 9)), {}),
        (changed_wav(32, struct.pack("<H", 4)), {}),
        (changed_wav(28, struct.pack("<I", 24000)), {}),
        (resized_riff(wav_bytes() + b"x"), {}),
        (resized_riff(wav_bytes() + b"JUNK" + struct.pack("<I", 5)), {}),
        (resized_riff(wav_bytes(b"\x00" * 49) + b"\x00"), {}),
        (resized_riff(wav_bytes()[:-2]), {}),
        (wav_bytes(), {"content-length": str(101 * 1024 * 1024)}),
        (wav_bytes(), {"content-length": "invalid"}),
        (wav_bytes(), {"content-encoding": "unknown"}),
    ],
)
def test_rejects_invalid_or_unbounded_audio(invoke, content, headers):
    responses = iter([httpx.Response(200, json=job()), httpx.Response(200, content=content, headers=headers)])
    with pytest.raises(FlowSpeechException):
        invoke(handler(Clock()), lambda request: next(responses))


def test_accepts_wav_with_padded_metadata_chunk(invoke):
    audio = wav_bytes(metadata=True)
    responses = iter([httpx.Response(200, json=job()), httpx.Response(200, content=audio)])
    assert invoke(handler(Clock()), lambda request: next(responses)).content == audio


def test_stream_size_limit_without_content_length(invoke):
    tts = handler(Clock())
    tts.MAX_AUDIO_BYTES = 10
    responses = iter([httpx.Response(200, json=job()), httpx.Response(200, content=wav_bytes())])

    def respond(request):
        response = next(responses)
        response.headers.pop("content-length", None)
        return response

    with pytest.raises(FlowSpeechException, match="size limit"):
        invoke(tts, respond)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("successful", [False, True])
@pytest.mark.asyncio
async def test_client_lifetime(monkeypatch, asynchronous, owned, successful):
    methods = []

    def respond(request):
        methods.append(request.method)
        if request.method == "POST":
            return httpx.Response(202, json=job("success" if successful else "failed"))
        return httpx.Response(200, content=wav_bytes())

    client_type = httpx.AsyncClient if asynchronous else httpx.Client
    client = client_type(transport=httpx.MockTransport(respond))

    def create_client(*, trust_env):
        assert trust_env is False
        return client

    if owned:
        if asynchronous:
            get_async_httpx_client(LlmProviders.FLOWSPEECH).client = client
        else:
            monkeypatch.setattr(httpx, "Client", create_client)
    tts = handler(Clock())

    async def call():
        if asynchronous:
            return await tts.arun(None if owned else client)
        return tts.run(None if owned else client)

    try:
        if successful:
            result = await call()
            assert result.content == wav_bytes()
        else:
            with pytest.raises(FlowSpeechException, match="FlowSpeech job failed"):
                await call()
        assert methods == (["POST", "GET"] if successful else ["POST"])
        assert client.is_closed is (owned and not asynchronous)
    finally:
        if asynchronous:
            await client.aclose()
        else:
            client.close()


@pytest.mark.asyncio
async def test_default_async_client_is_reused_without_download_credentials():
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(202, json=job())
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        assert "x-client-secret" not in request.headers
        return httpx.Response(200, content=wav_bytes())

    cached = get_async_httpx_client(LlmProviders.FLOWSPEECH)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        headers={"Authorization": "discard", "X-Client-Secret": "discard"},
        cookies={"session": "discard"},
    ) as client:
        cached.client = client
        for _ in range(2):
            result = await handler(Clock()).arun()
            assert result.content == wav_bytes()
            assert not client.is_closed
        assert get_async_httpx_client(LlmProviders.FLOWSPEECH).client is client
    assert [request.method for request in requests] == ["POST", "GET", "POST", "GET"]


@pytest.mark.asyncio
async def test_async_total_deadline_interrupts_network():
    async def respond(request):
        await asyncio.sleep(1)
        return httpx.Response(202, json=job("queued"))

    tts = FlowSpeechTTS(URL, {"Idempotency-Key": REQUEST_ID}, {"input": {}}, timeout=0.02)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(FlowSpeechException, match="total deadline"):
            await tts.arun(client)


@pytest.mark.asyncio
async def test_async_cancellation_propagates():
    started = asyncio.Event()

    async def respond(request):
        started.set()
        await asyncio.sleep(100)
        return httpx.Response(200, json=job())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        task = asyncio.create_task(handler(Clock()).arun(client))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
