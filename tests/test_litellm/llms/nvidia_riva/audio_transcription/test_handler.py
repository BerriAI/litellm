"""
End-to-end-ish tests for NvidiaRivaAudioTranscription.

We mock ``riva.client`` so the test does not need the real gRPC SDK or a
running Riva server. The mock also lets us assert how Auth metadata is
constructed (NVCF vs self-hosted) and how the streaming generator output
is aggregated.
"""

import asyncio
import io
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.nvidia_riva.audio_transcription import handler as handler_mod
from litellm.llms.nvidia_riva.audio_transcription.handler import (
    NvidiaRivaAudioTranscription,
)
from litellm.llms.nvidia_riva.common_utils import NvidiaRivaException
from litellm.types.utils import TranscriptionResponse


def _make_wav_bytes(seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    n = int(sample_rate * seconds)
    samples = (0.05 * np.sin(np.linspace(0, 2 * np.pi * 220 * seconds, n))).astype(
        np.float32
    )
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _fake_word(word: str, start_ms: int, end_ms: int):
    return SimpleNamespace(word=word, start_time=start_ms, end_time=end_ms)


def _fake_alternative(transcript: str, words=None):
    return SimpleNamespace(transcript=transcript, words=words or [])


def _fake_result(is_final: bool, alternatives):
    return SimpleNamespace(is_final=is_final, alternatives=alternatives)


def _fake_response(results):
    return SimpleNamespace(results=results)


@pytest.fixture
def mock_riva(monkeypatch):
    """
    Stand-ins for the bits of ``riva.client`` the handler touches:
    - ``Auth`` (constructor)
    - ``ASRService`` with ``streaming_response_generator``
    - ``RecognitionConfig``, ``StreamingRecognitionConfig``, ``EndpointingConfig``
    - ``AudioEncoding`` namespace with ``LINEAR_PCM``
    """
    auth_calls = {}

    class FakeAuth:
        def __init__(self, *args, **kwargs):
            # Support both keyword and positional Auth constructors.
            if kwargs:
                auth_calls["uri"] = kwargs.get("uri")
                auth_calls["use_ssl"] = kwargs.get("use_ssl")
                auth_calls["metadata_args"] = kwargs.get("metadata_args")
            else:
                # positional: (None, use_ssl, uri, metadata)
                auth_calls["use_ssl"] = args[1] if len(args) > 1 else None
                auth_calls["uri"] = args[2] if len(args) > 2 else None
                auth_calls["metadata_args"] = args[3] if len(args) > 3 else None

    class FakeRecognitionConfig:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            self.endpointing_config = SimpleNamespace(CopyFrom=lambda _: None)

    class FakeStreamingRecognitionConfig:
        def __init__(self, config, interim_results):
            self.config = config
            self.interim_results = interim_results

    class FakeEndpointingConfig:
        def __init__(self, **kwargs):
            self._kwargs = kwargs

    class FakeAudioEncoding:
        LINEAR_PCM = "LINEAR_PCM"

    streaming_responses_holder = {"value": []}

    class FakeASRService:
        def __init__(self, auth):
            self.auth = auth

        def streaming_response_generator(self, audio_chunks, streaming_config):
            # Drain audio_chunks generator so we exercise the chunking path.
            list(audio_chunks)
            yield from streaming_responses_holder["value"]

    fake_riva_client = SimpleNamespace(
        Auth=FakeAuth,
        ASRService=FakeASRService,
        RecognitionConfig=FakeRecognitionConfig,
        StreamingRecognitionConfig=FakeStreamingRecognitionConfig,
        EndpointingConfig=FakeEndpointingConfig,
        AudioEncoding=FakeAudioEncoding,
    )

    def fake_import_riva():
        return fake_riva_client, fake_riva_client

    monkeypatch.setattr(handler_mod, "_import_riva", fake_import_riva)

    return SimpleNamespace(
        auth_calls=auth_calls,
        responses=streaming_responses_holder,
        client=fake_riva_client,
    )


@pytest.fixture
def logging_obj():
    return MagicMock()


def test_sync_path_aggregates_only_final_results(mock_riva, logging_obj):
    mock_riva.responses["value"] = [
        # Empty heartbeat chunk: ignore.
        _fake_response(results=[]),
        # Interim chunk (not final): ignore.
        _fake_response(
            results=[
                _fake_result(
                    is_final=False, alternatives=[_fake_alternative("partial...")]
                )
            ]
        ),
        # Two final chunks aggregated.
        _fake_response(
            results=[
                _fake_result(
                    is_final=True,
                    alternatives=[
                        _fake_alternative(
                            "Hello,",
                            words=[_fake_word("Hello,", 0, 320)],
                        )
                    ],
                )
            ]
        ),
        _fake_response(
            results=[
                _fake_result(
                    is_final=True,
                    alternatives=[
                        _fake_alternative(
                            " world.",
                            words=[_fake_word("world.", 480, 870)],
                        )
                    ],
                )
            ]
        ),
    ]

    impl = NvidiaRivaAudioTranscription()
    response: TranscriptionResponse = impl.audio_transcriptions(
        model="nvidia/parakeet-ctc-1_1b-asr",
        audio_file=_make_wav_bytes(),
        optional_params={
            "language_code": "en-US",
            "enable_word_time_offsets": True,
            "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
        },
        litellm_params={},
        model_response=TranscriptionResponse(),
        timeout=60,
        logging_obj=logging_obj,
        api_key="nvapi-xxx",
        api_base="grpc.nvcf.nvidia.com:443",
    )

    assert response.text == "Hello, world."
    # duration is propagated from the resampler.
    assert response._hidden_params["audio_transcription_duration"] == pytest.approx(
        1.0, abs=0.05
    )
    # word timestamps converted from ms to seconds.
    words = response["words"]
    assert words[0]["start"] == pytest.approx(0.0)
    assert words[1]["end"] == pytest.approx(0.87)
    assert (
        logging_obj.pre_call.call_args.kwargs["additional_args"]["atranscription"]
        is False
    )


def test_auth_nvcf_defaults_use_ssl_and_attaches_function_id(mock_riva, logging_obj):
    mock_riva.responses["value"] = [
        _fake_response(
            results=[
                _fake_result(
                    is_final=True,
                    alternatives=[_fake_alternative("ok")],
                )
            ]
        )
    ]
    impl = NvidiaRivaAudioTranscription()
    impl.audio_transcriptions(
        model="m",
        audio_file=_make_wav_bytes(),
        optional_params={
            "nvcf_function_id": "abc-123",
            "language_code": "en-US",
        },
        litellm_params={},
        model_response=TranscriptionResponse(),
        timeout=60,
        logging_obj=logging_obj,
        api_key="nvapi-xxx",
        api_base="grpc.nvcf.nvidia.com:443",
    )

    assert mock_riva.auth_calls["uri"] == "grpc.nvcf.nvidia.com:443"
    assert mock_riva.auth_calls["use_ssl"] is True
    metadata = dict(mock_riva.auth_calls["metadata_args"])
    assert metadata["function-id"] == "abc-123"
    assert metadata["authorization"] == "Bearer nvapi-xxx"


def test_auth_self_hosted_defaults_no_ssl_and_no_function_id(mock_riva, logging_obj):
    mock_riva.responses["value"] = [
        _fake_response(
            results=[
                _fake_result(is_final=True, alternatives=[_fake_alternative("ok")])
            ]
        )
    ]
    impl = NvidiaRivaAudioTranscription()
    impl.audio_transcriptions(
        model="m",
        audio_file=_make_wav_bytes(),
        optional_params={"language_code": "en-US"},
        litellm_params={},
        model_response=TranscriptionResponse(),
        timeout=60,
        logging_obj=logging_obj,
        api_key=None,
        api_base="localhost:50051",
    )

    assert mock_riva.auth_calls["uri"] == "localhost:50051"
    assert mock_riva.auth_calls["use_ssl"] is False
    metadata = dict(mock_riva.auth_calls["metadata_args"])
    # No function-id, no authorization metadata.
    assert "function-id" not in metadata
    assert "authorization" not in metadata


def test_explicit_use_ssl_override_wins(mock_riva, logging_obj):
    """
    Self-hosted Riva behind an ingress with TLS termination is a real
    deployment topology. ``use_ssl=True`` must be honored even without an
    NVCF function id.
    """
    mock_riva.responses["value"] = [
        _fake_response(
            results=[
                _fake_result(is_final=True, alternatives=[_fake_alternative("ok")])
            ]
        )
    ]
    impl = NvidiaRivaAudioTranscription()
    impl.audio_transcriptions(
        model="m",
        audio_file=_make_wav_bytes(),
        optional_params={"use_ssl": True, "language_code": "en-US"},
        litellm_params={},
        model_response=TranscriptionResponse(),
        timeout=60,
        logging_obj=logging_obj,
        api_key=None,
        api_base="riva.internal.company.com:443",
    )

    assert mock_riva.auth_calls["use_ssl"] is True


def test_missing_api_base_raises_clear_error(mock_riva, logging_obj):
    impl = NvidiaRivaAudioTranscription()
    with pytest.raises(NvidiaRivaException) as excinfo:
        impl.audio_transcriptions(
            model="m",
            audio_file=_make_wav_bytes(),
            optional_params={},
            litellm_params={},
            model_response=TranscriptionResponse(),
            timeout=60,
            logging_obj=logging_obj,
            api_key=None,
            api_base=None,
        )
    assert "api_base" in excinfo.value.message


def test_async_path_uses_to_thread(mock_riva, logging_obj):
    mock_riva.responses["value"] = [
        _fake_response(
            results=[
                _fake_result(
                    is_final=True, alternatives=[_fake_alternative("async ok")]
                )
            ]
        )
    ]
    impl = NvidiaRivaAudioTranscription()
    response = asyncio.run(
        impl.async_audio_transcriptions(
            model="m",
            audio_file=_make_wav_bytes(),
            optional_params={"language_code": "en-US"},
            litellm_params={},
            model_response=TranscriptionResponse(),
            timeout=60,
            logging_obj=logging_obj,
            api_key=None,
            api_base="localhost:50051",
        )
    )
    assert response.text == "async ok"
    assert (
        logging_obj.pre_call.call_args.kwargs["additional_args"]["atranscription"]
        is True
    )


def test_timeout_is_forwarded_to_streaming_generator_when_supported(
    mock_riva, logging_obj
):
    """
    Without a deadline the gRPC stream can block forever on a stalled Riva
    server. The handler must forward the call-level ``timeout`` to
    ``streaming_response_generator`` whenever the installed riva-client
    accepts a ``timeout`` kwarg.
    """
    captured_kwargs = {}

    def streaming_with_timeout(self, audio_chunks, streaming_config, timeout=None):
        captured_kwargs["timeout"] = timeout
        list(audio_chunks)
        yield from [
            _fake_response(
                results=[
                    _fake_result(is_final=True, alternatives=[_fake_alternative("ok")])
                ]
            )
        ]

    mock_riva.client.ASRService.streaming_response_generator = streaming_with_timeout

    impl = NvidiaRivaAudioTranscription()
    impl.audio_transcriptions(
        model="m",
        audio_file=_make_wav_bytes(),
        optional_params={"language_code": "en-US"},
        litellm_params={},
        model_response=TranscriptionResponse(),
        timeout=12.5,
        logging_obj=logging_obj,
        api_key=None,
        api_base="localhost:50051",
    )
    assert captured_kwargs["timeout"] == pytest.approx(12.5)


def test_grpc_error_is_wrapped_as_nvidia_riva_exception(mock_riva, logging_obj):
    class FakeGrpcError(Exception):
        def code(self):
            return SimpleNamespace(name="UNAUTHENTICATED")

        def details(self):
            return "bad token"

    def raising_streaming_response_generator(self, audio_chunks, streaming_config):
        list(audio_chunks)
        raise FakeGrpcError("rpc fail")

    mock_riva.client.ASRService.streaming_response_generator = (
        raising_streaming_response_generator
    )

    impl = NvidiaRivaAudioTranscription()
    with pytest.raises(NvidiaRivaException) as excinfo:
        impl.audio_transcriptions(
            model="m",
            audio_file=_make_wav_bytes(),
            optional_params={"language_code": "en-US"},
            litellm_params={},
            model_response=TranscriptionResponse(),
            timeout=60,
            logging_obj=logging_obj,
            api_key="nvapi-xxx",
            api_base="grpc.nvcf.nvidia.com:443",
        )

    assert excinfo.value.status_code == 401
    assert "UNAUTHENTICATED" in excinfo.value.message
