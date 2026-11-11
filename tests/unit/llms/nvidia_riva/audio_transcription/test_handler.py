import asyncio
import io
import ipaddress
import threading
import uuid
import wave
from array import array
from collections.abc import Iterable, Iterator
from concurrent import futures
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from queue import SimpleQueue
from typing import Final

import grpc
import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from riva.client.proto import riva_asr_pb2, riva_asr_pb2_grpc, riva_audio_pb2

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.nvidia_riva.audio_transcription.handler import NvidiaRivaAudioTranscription
from litellm.llms.nvidia_riva.common_utils import NvidiaRivaException
from litellm.types.utils import TranscriptionResponse

_SAMPLE_RATE_HZ: Final = 16000
_SERVER_WAIT_SECONDS: Final = 30
_STALL_LIMIT_SECONDS: Final = 60
_NO_DEADLINE_SECONDS: Final = 10**9
_STALL_TIMEOUT_SECONDS: Final = 0.5
_API_KEY: Final = "nvapi-test"


@dataclass(frozen=True, slots=True)
class _Script:
    streaming: tuple[riva_asr_pb2.StreamingRecognizeResponse, ...] = ()
    offline: riva_asr_pb2.RecognizeResponse = field(default_factory=riva_asr_pb2.RecognizeResponse)
    abort: tuple[grpc.StatusCode, str] | None = None
    stall: bool = False


@dataclass(frozen=True, slots=True)
class _Call:
    rpc: str
    metadata: tuple[tuple[str, str], ...]
    config: riva_asr_pb2.RecognitionConfig
    audio: bytes
    deadline_seconds: float | None
    cancelled: bool

    def header(self, key: str) -> tuple[str, ...]:
        return tuple(value for name, value in self.metadata if name == key)


@dataclass(frozen=True, slots=True)
class _Riva:
    address: str
    calls: SimpleQueue[_Call]

    def only_call(self) -> _Call:
        call: Final = self.calls.get(timeout=_SERVER_WAIT_SECONDS)
        assert self.calls.empty(), "Riva received more than one call"
        return call


class _Asr(riva_asr_pb2_grpc.RivaSpeechRecognitionServicer):
    def __init__(self, script: _Script, calls: SimpleQueue[_Call]) -> None:
        self._script: Final = script
        self._calls: Final = calls

    def _settle(
        self, context: grpc.ServicerContext, rpc: str, config: riva_asr_pb2.RecognitionConfig, audio: bytes
    ) -> None:
        remaining: Final = context.time_remaining()
        released: Final = threading.Event()
        if self._script.stall and context.add_callback(released.set):
            released.wait(timeout=_STALL_LIMIT_SECONDS)
        self._calls.put(
            _Call(
                rpc=rpc,
                metadata=tuple((item.key, item.value) for item in context.invocation_metadata()),
                config=config,
                audio=audio,
                deadline_seconds=None if remaining is None or remaining > _NO_DEADLINE_SECONDS else remaining,
                cancelled=not context.is_active(),
            )
        )
        if self._script.abort is not None:
            context.abort(*self._script.abort)

    def Recognize(
        self, request: riva_asr_pb2.RecognizeRequest, context: grpc.ServicerContext
    ) -> riva_asr_pb2.RecognizeResponse:
        self._settle(context, "Recognize", request.config, request.audio)
        return self._script.offline

    def StreamingRecognize(
        self, request_iterator: Iterable[riva_asr_pb2.StreamingRecognizeRequest], context: grpc.ServicerContext
    ) -> Iterator[riva_asr_pb2.StreamingRecognizeResponse]:
        first, *rest = tuple(request_iterator)
        self._settle(
            context, "StreamingRecognize", first.streaming_config.config, b"".join(r.audio_content for r in rest)
        )
        yield from self._script.streaming


@dataclass(frozen=True, slots=True)
class _Tls:
    key_pem: bytes
    cert_pem: bytes


def _self_signed_loopback_tls() -> _Tls:
    key: Final = ec.generate_private_key(ec.SECP256R1())
    name: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert: Final = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime(2000, 1, 1, tzinfo=timezone.utc))
        .not_valid_after(datetime(2099, 1, 1, tzinfo=timezone.utc))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    return _Tls(
        key_pem=key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        ),
        cert_pem=cert.public_bytes(serialization.Encoding.PEM),
    )


@contextmanager
def _riva_server(script: _Script, max_receive_bytes: int | None = None, tls: _Tls | None = None) -> Iterator[_Riva]:
    calls: Final[SimpleQueue[_Call]] = SimpleQueue()
    options: Final = () if max_receive_bytes is None else (("grpc.max_receive_message_length", max_receive_bytes),)
    server: Final = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=options)
    riva_asr_pb2_grpc.add_RivaSpeechRecognitionServicer_to_server(_Asr(script, calls), server)
    port: Final = (
        server.add_insecure_port("127.0.0.1:0")
        if tls is None
        else server.add_secure_port("127.0.0.1:0", grpc.ssl_server_credentials(((tls.key_pem, tls.cert_pem),)))
    )
    server.start()
    try:
        yield _Riva(f"127.0.0.1:{port}", calls)
    finally:
        server.stop(grace=None).wait()


def _samples(seconds: float) -> array[int]:
    return array("h", (((index * 37) % 2001 - 1000) * 16 for index in range(int(seconds * _SAMPLE_RATE_HZ))))


def _wav(seconds: float = 1.0) -> bytes:
    buffer: Final = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(_SAMPLE_RATE_HZ)
        writer.writeframes(_samples(seconds).tobytes())
    return buffer.getvalue()


def _carries_clip(audio: bytes, seconds: float) -> bool:
    sent: Final = _samples(seconds)
    if len(audio) != len(sent) * 2:
        return False
    return all(abs(left - right) <= 1 for left, right in zip(sent, array("h", audio), strict=True))


def _alternative(
    transcript: str, words: tuple[tuple[str, int, int], ...] = ()
) -> riva_asr_pb2.SpeechRecognitionAlternative:
    return riva_asr_pb2.SpeechRecognitionAlternative(
        transcript=transcript,
        words=[riva_asr_pb2.WordInfo(word=word, start_time=start, end_time=end) for word, start, end in words],
    )


def _streamed(transcript: str, is_final: bool = True) -> riva_asr_pb2.StreamingRecognizeResponse:
    return riva_asr_pb2.StreamingRecognizeResponse(
        results=[riva_asr_pb2.StreamingRecognitionResult(alternatives=[_alternative(transcript)], is_final=is_final)]
    )


def _offline(*transcripts: str) -> riva_asr_pb2.RecognizeResponse:
    return riva_asr_pb2.RecognizeResponse(
        results=[riva_asr_pb2.SpeechRecognitionResult(alternatives=[_alternative(text)]) for text in transcripts]
    )


def _answers(transcript: str) -> _Script:
    return _Script(streaming=(_streamed(transcript),), offline=_offline(transcript))


def _logging() -> Logging:
    return Logging(
        model="m",
        messages=[],
        stream=False,
        call_type="transcription",
        start_time=datetime(2026, 1, 1),
        litellm_call_id=str(uuid.uuid4()),
        function_id="riva-test",
    )


def _transcribe(
    riva: _Riva,
    optional_params: dict[str, object] | None = None,
    *,
    api_key: str | None = None,
    timeout: float | str | httpx.Timeout | None = 60,
    litellm_params: dict[str, object] | None = None,
) -> TranscriptionResponse:
    return NvidiaRivaAudioTranscription().audio_transcriptions(
        model="m",
        audio_file=_wav(),
        optional_params={"language_code": "en-US", **(optional_params or {})},
        litellm_params=litellm_params or {},
        model_response=TranscriptionResponse(),
        timeout=timeout,
        logging_obj=_logging(),
        api_key=api_key,
        api_base=riva.address,
    )


def test_streaming_aggregates_only_final_results_and_their_word_timings() -> None:
    heartbeat: Final = riva_asr_pb2.StreamingRecognizeResponse()
    hello: Final = riva_asr_pb2.StreamingRecognizeResponse(
        results=[
            riva_asr_pb2.StreamingRecognitionResult(
                alternatives=[_alternative("Hello,", (("Hello,", 0, 320),))], is_final=True
            )
        ]
    )
    world: Final = riva_asr_pb2.StreamingRecognizeResponse(
        results=[
            riva_asr_pb2.StreamingRecognitionResult(
                alternatives=[_alternative(" world.", (("world.", 480, 870),))], is_final=True
            )
        ]
    )
    with _riva_server(_Script(streaming=(heartbeat, _streamed("partial...", is_final=False), hello, world))) as riva:
        response: Final = _transcribe(
            riva,
            {"enable_word_time_offsets": True, "response_format": "verbose_json", "timestamp_granularities": ["word"]},
        )
        call: Final = riva.only_call()

    assert response.text == "Hello, world."
    assert response["words"] == [
        {"word": "Hello,", "start": 0.0, "end": 0.32},
        {"word": "world.", "start": 0.48, "end": 0.87},
    ]
    assert response._hidden_params["audio_transcription_duration"] == pytest.approx(1.0)
    assert call.rpc == "StreamingRecognize"
    assert _carries_clip(call.audio, 1.0), "streamed chunks do not add up to the resampled clip"


def test_riva_offline_uses_recognize_and_keeps_results_without_is_final() -> None:
    offline: Final = riva_asr_pb2.RecognizeResponse(
        results=[
            riva_asr_pb2.SpeechRecognitionResult(alternatives=[_alternative("Hello,", (("Hello,", 0, 320),))]),
            riva_asr_pb2.SpeechRecognitionResult(alternatives=[_alternative(" world.", (("world.", 480, 870),))]),
        ]
    )
    with _riva_server(_Script(offline=offline)) as riva:
        response: Final = _transcribe(
            riva,
            {
                "language_code": "multi",
                "riva_offline": True,
                "enable_word_time_offsets": True,
                "response_format": "verbose_json",
                "timestamp_granularities": ["word"],
            },
        )
        call: Final = riva.only_call()

    assert response.text == "Hello, world."
    assert response["words"][1] == {"word": "world.", "start": 0.48, "end": 0.87}
    assert response._hidden_params["audio_transcription_duration"] == pytest.approx(1.0)
    assert call.rpc == "Recognize"
    assert call.config.language_code == "multi"
    assert _carries_clip(call.audio, 1.0), "offline request does not carry the whole resampled clip"


@pytest.mark.parametrize("riva_offline", [False, True])
def test_recognition_config_reaches_riva_as_nvidia_decodes_it(riva_offline: bool) -> None:
    with _riva_server(_answers("ok")) as riva:
        _transcribe(
            riva,
            {
                "riva_offline": riva_offline,
                "enable_word_time_offsets": True,
                "riva_model_name": "parakeet",
                "sample_rate_hertz": 8000,
                "profanity_filter": True,
            },
        )
        call: Final = riva.only_call()

    assert call.config == riva_asr_pb2.RecognitionConfig(
        encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=_SAMPLE_RATE_HZ,
        language_code="en-US",
        max_alternatives=1,
        profanity_filter=True,
        audio_channel_count=1,
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
        model="parakeet",
    )


def test_default_mode_streams_and_never_calls_recognize() -> None:
    with _riva_server(_answers("streamed")) as riva:
        response: Final = _transcribe(riva)
        call: Final = riva.only_call()

    assert response.text == "streamed"
    assert call.rpc == "StreamingRecognize"


@pytest.mark.parametrize(
    ("riva_offline", "rpc"), [("false", "StreamingRecognize"), ("False", "StreamingRecognize"), ("true", "Recognize")]
)
def test_riva_offline_string_from_multipart_form_is_parsed(riva_offline: str, rpc: str) -> None:
    with _riva_server(_Script(streaming=(_streamed("streamed"),), offline=_offline("offline"))) as riva:
        response: Final = _transcribe(riva, {"riva_offline": riva_offline})
        call: Final = riva.only_call()

    assert call.rpc == rpc
    assert response.text == ("offline" if rpc == "Recognize" else "streamed")


@pytest.mark.parametrize("riva_offline", ["yes", "", 1])
def test_riva_offline_malformed_value_is_rejected_before_any_rpc(riva_offline: object) -> None:
    with _riva_server(_answers("ok")) as riva:
        with pytest.raises(NvidiaRivaException) as excinfo:
            _transcribe(riva, {"riva_offline": riva_offline})

        assert riva.calls.empty()

    assert excinfo.value.status_code == 400
    assert "riva_offline" in excinfo.value.message


def test_riva_offline_malformed_value_is_dropped_under_drop_params() -> None:
    with _riva_server(_Script(streaming=(_streamed("streamed"),), offline=_offline("offline"))) as riva:
        response: Final = _transcribe(riva, {"riva_offline": "yes"}, litellm_params={"drop_params": True})
        call: Final = riva.only_call()

    assert response.text == "streamed"
    assert call.rpc == "StreamingRecognize"


def test_deployment_riva_offline_reaches_the_handler_through_transcription() -> None:
    with _riva_server(_Script(offline=_offline("offline"))) as riva:
        response: Final = litellm.transcription(
            model="nvidia_riva/nvidia/parakeet-0.6b-tdt",
            file=("clip.wav", _wav(), "audio/wav"),
            api_base=riva.address,
            riva_offline=True,
        )
        call: Final = riva.only_call()

    assert response.text == "offline"
    assert call.rpc == "Recognize"


@pytest.mark.parametrize(
    ("timeout", "deadline_seconds"),
    [
        (12.5, 12.5),
        ("12.5", 12.5),
        (httpx.Timeout(60.0, read=12.5), 12.5),
        (httpx.Timeout(5.0, read=None), 600.0),
        (None, None),
    ],
)
def test_streaming_deadline_is_the_timeout_in_seconds_whatever_its_form(
    timeout: float | str | httpx.Timeout | None, deadline_seconds: float | None
) -> None:
    with _riva_server(_answers("streamed")) as riva:
        response: Final = litellm.transcription(
            model="nvidia_riva/nvidia/parakeet-ctc-1.1b",
            file=("clip.wav", _wav(), "audio/wav"),
            api_base=riva.address,
            timeout=timeout,
        )
        call: Final = riva.only_call()

    assert response.text == "streamed"
    assert call.deadline_seconds == (None if deadline_seconds is None else pytest.approx(deadline_seconds, rel=0.25))


@pytest.mark.parametrize("riva_offline", [False, True])
@pytest.mark.parametrize("timeout", ["abc", "30s"])
def test_malformed_timeout_is_a_400_before_any_rpc(riva_offline: bool, timeout: str) -> None:
    with _riva_server(_answers("ok")) as riva:
        with pytest.raises(NvidiaRivaException) as excinfo:
            litellm.transcription(
                model="nvidia_riva/nvidia/parakeet-0.6b-tdt",
                file=("clip.wav", _wav(), "audio/wav"),
                api_base=riva.address,
                riva_offline=riva_offline,
                timeout=timeout,
            )

        assert riva.calls.empty()

    assert excinfo.value.status_code == 400
    assert "timeout" in excinfo.value.message


def test_malformed_streaming_timeout_falls_back_to_the_default_deadline_under_drop_params() -> None:
    with _riva_server(_answers("ok")) as riva:
        response: Final = litellm.transcription(
            model="nvidia_riva/nvidia/parakeet-ctc-1.1b",
            file=("clip.wav", _wav(), "audio/wav"),
            api_base=riva.address,
            timeout="abc",
            drop_params=True,
        )
        call: Final = riva.only_call()

    assert response.text == "ok"
    assert call.rpc == "StreamingRecognize"
    assert call.deadline_seconds == pytest.approx(600.0, rel=0.25)


def test_malformed_offline_timeout_is_ignored_under_drop_params() -> None:
    with _riva_server(_answers("ok")) as riva:
        response: Final = litellm.transcription(
            model="nvidia_riva/nvidia/parakeet-0.6b-tdt",
            file=("clip.wav", _wav(), "audio/wav"),
            api_base=riva.address,
            riva_offline=True,
            timeout="abc",
            drop_params=True,
        )
        call: Final = riva.only_call()

    assert response.text == "ok"
    assert call.rpc == "Recognize"


@pytest.mark.parametrize(
    "timeout", [_STALL_TIMEOUT_SECONDS, str(_STALL_TIMEOUT_SECONDS), httpx.Timeout(30.0, read=_STALL_TIMEOUT_SECONDS)]
)
def test_stalled_offline_recognition_is_cancelled_and_times_out_as_408(timeout: float | str | httpx.Timeout) -> None:
    with _riva_server(_Script(stall=True)) as riva:
        with pytest.raises(NvidiaRivaException) as excinfo:
            _transcribe(riva, {"riva_offline": True}, timeout=timeout)
        call: Final = riva.only_call()

    assert excinfo.value.status_code == 408
    assert f"within {_STALL_TIMEOUT_SECONDS} seconds" in excinfo.value.message
    assert call.cancelled is True


def test_stalled_streaming_recognition_hits_its_deadline_as_504() -> None:
    with _riva_server(_Script(stall=True)) as riva:
        with pytest.raises(NvidiaRivaException) as excinfo:
            _transcribe(riva, timeout=_STALL_TIMEOUT_SECONDS)
        call: Final = riva.only_call()

    assert excinfo.value.status_code == 504, excinfo.value.message
    assert "DEADLINE_EXCEEDED" in excinfo.value.message
    assert call.cancelled is True


def test_oversized_offline_audio_is_a_non_retryable_413() -> None:
    with _riva_server(_answers("ok"), max_receive_bytes=4096) as riva:
        with pytest.raises(NvidiaRivaException) as excinfo:
            _transcribe(riva, {"riva_offline": True})

        assert riva.calls.empty()

    assert excinfo.value.status_code == 413, excinfo.value.message
    assert litellm.utils.should_retry(excinfo.value.status_code) is False


@pytest.mark.parametrize(
    ("riva_offline", "code", "status_code"),
    [
        (False, grpc.StatusCode.UNAUTHENTICATED, 401),
        (True, grpc.StatusCode.INVALID_ARGUMENT, 400),
        (False, grpc.StatusCode.RESOURCE_EXHAUSTED, 429),
    ],
)
def test_grpc_error_is_wrapped_as_nvidia_riva_exception(
    riva_offline: bool, code: grpc.StatusCode, status_code: int
) -> None:
    with _riva_server(_Script(abort=(code, "Riva said no"))) as riva:
        with pytest.raises(NvidiaRivaException) as excinfo:
            _transcribe(riva, {"riva_offline": riva_offline}, api_key=_API_KEY)

    assert excinfo.value.status_code == status_code
    assert code.name in excinfo.value.message
    assert "Riva said no" in excinfo.value.message


@pytest.mark.parametrize("riva_offline", [False, True])
def test_self_hosted_api_key_is_sent_as_one_bearer_token(riva_offline: bool) -> None:
    with _riva_server(_answers("ok")) as riva:
        _transcribe(riva, {"riva_offline": riva_offline}, api_key=_API_KEY)
        call: Final = riva.only_call()

    assert call.header("authorization") == (f"Bearer {_API_KEY}",)
    assert call.header("function-id") == ()


@pytest.mark.parametrize("riva_offline", [False, True])
def test_nvcf_function_id_and_api_key_are_each_sent_once(riva_offline: bool) -> None:
    with _riva_server(_answers("ok")) as riva:
        _transcribe(
            riva, {"riva_offline": riva_offline, "nvcf_function_id": "abc-123", "use_ssl": False}, api_key=_API_KEY
        )
        call: Final = riva.only_call()

    assert call.header("function-id") == ("abc-123",)
    assert call.header("authorization") == (f"Bearer {_API_KEY}",)


@pytest.mark.parametrize("riva_offline", [False, True])
def test_no_api_key_sends_no_authorization(riva_offline: bool) -> None:
    with _riva_server(_answers("ok")) as riva:
        _transcribe(riva, {"riva_offline": riva_offline})
        call: Final = riva.only_call()

    assert call.header("authorization") == ()
    assert call.header("function-id") == ()


@pytest.mark.parametrize("riva_offline", [False, True])
@pytest.mark.parametrize(
    ("optional_params", "function_ids"),
    [({"use_ssl": True}, ()), ({"nvcf_function_id": "abc-123"}, ("abc-123",))],
    ids=["use_ssl", "nvcf_default_tls"],
)
def test_tls_api_key_is_sent_as_one_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
    optional_params: dict[str, object],
    function_ids: tuple[str, ...],
    riva_offline: bool,
) -> None:
    tls: Final = _self_signed_loopback_tls()
    system_roots: Final = grpc.ssl_channel_credentials
    monkeypatch.setattr(grpc, "ssl_channel_credentials", lambda: system_roots(root_certificates=tls.cert_pem))
    with _riva_server(_answers("ok"), tls=tls) as riva:
        response: Final = _transcribe(riva, {**optional_params, "riva_offline": riva_offline}, api_key=_API_KEY)
        call: Final = riva.only_call()

    assert response.text == "ok"
    assert call.header("authorization") == (f"Bearer {_API_KEY}",)
    assert call.header("function-id") == function_ids


@pytest.mark.parametrize("optional_params", [{"use_ssl": True}, {"nvcf_function_id": "abc-123"}])
def test_tls_never_falls_back_to_plaintext(optional_params: dict[str, object]) -> None:
    with _riva_server(_answers("ok")) as riva:
        with pytest.raises(NvidiaRivaException) as excinfo:
            _transcribe(riva, optional_params, api_key=_API_KEY, timeout=_SERVER_WAIT_SECONDS)

        assert riva.calls.empty()

    assert excinfo.value.status_code == 503, excinfo.value.message


def test_explicit_endpointing_config_keeps_zero_thresholds() -> None:
    with _riva_server(_answers("ok")) as riva:
        _transcribe(riva, {"endpointing_config": {"stop_threshold": 0.0, "stop_history": 800}})
        call: Final = riva.only_call()

    assert call.config.HasField("endpointing_config")
    assert call.config.endpointing_config.HasField("stop_threshold")
    assert call.config.endpointing_config == riva_asr_pb2.EndpointingConfig(stop_threshold=0.0, stop_history=800)


def test_unknown_endpointing_field_falls_back_to_riva_defaults() -> None:
    with _riva_server(_answers("ok")) as riva:
        response: Final = _transcribe(riva, {"endpointing_config": {"stop_threshold": 0.5, "not_a_field": 1}})
        call: Final = riva.only_call()

    assert response.text == "ok"
    assert not call.config.HasField("endpointing_config")


def test_missing_api_base_raises_clear_error() -> None:
    with pytest.raises(NvidiaRivaException) as excinfo:
        NvidiaRivaAudioTranscription().audio_transcriptions(
            model="m",
            audio_file=_wav(),
            optional_params={},
            litellm_params={},
            model_response=TranscriptionResponse(),
            timeout=60,
            logging_obj=_logging(),
            api_key=None,
            api_base=None,
        )

    assert excinfo.value.status_code == 400
    assert "api_base" in excinfo.value.message


def test_async_path_transcribes() -> None:
    with _riva_server(_answers("async ok")) as riva:
        response: Final = asyncio.run(
            NvidiaRivaAudioTranscription().async_audio_transcriptions(
                model="m",
                audio_file=_wav(),
                optional_params={"language_code": "en-US"},
                litellm_params={},
                model_response=TranscriptionResponse(),
                timeout=60,
                logging_obj=_logging(),
                api_key=_API_KEY,
                api_base=riva.address,
            )
        )
        call: Final = riva.only_call()

    assert response.text == "async ok"
    assert call.header("authorization") == (f"Bearer {_API_KEY}",)
