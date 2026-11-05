"""
gRPC calls to NVIDIA Riva's speech recognition service.

The ``authorization`` and NVCF ``function-id`` metadata ride on each call, never on the channel, so every
request carries them exactly once over both plaintext and TLS.
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Final, TypeAlias

import grpc
from google.protobuf.message import Message
from pydantic import TypeAdapter

from litellm.llms.nvidia_riva.audio_transcription import riva_asr_schema as schema
from litellm.llms.nvidia_riva.common_utils import NvidiaRivaException

# Stream audio to Riva in ~50 ms slices (1600 samples at 16 kHz). Matches
# NVIDIA's recommended chunk size for streaming ASR, small enough for
# responsive endpointing, large enough to keep per-RPC overhead low.
_CHUNK_SAMPLES: Final = 1600
_CHUNK_BYTES: Final = _CHUNK_SAMPLES * 2

CallMetadata: TypeAlias = tuple[tuple[str, str], ...]

_ENDPOINTING_FIELDS: Final = TypeAdapter(Mapping[str, object])


def call_metadata(api_key: str | None, nvcf_function_id: object) -> CallMetadata:
    function_id: Final = (("function-id", str(nvcf_function_id)),) if nvcf_function_id else ()
    authorization: Final = (("authorization", f"Bearer {api_key}"),) if api_key else ()
    return function_id + authorization


def open_channel(api_base: str, use_ssl: bool) -> grpc.Channel:
    if use_ssl:
        return grpc.secure_channel(api_base, grpc.ssl_channel_credentials())
    return grpc.insecure_channel(api_base)


def recognition_config(config: Mapping[str, Any]) -> Message:
    endpointing: Final = _endpointing_config(config.get("endpointing_config"))
    return schema.RecognitionConfig(
        encoding=schema.LINEAR_PCM,
        sample_rate_hertz=int(config["sample_rate_hertz"]),
        language_code=config["language_code"],
        audio_channel_count=int(config["audio_channel_count"]),
        enable_automatic_punctuation=bool(config.get("enable_automatic_punctuation", True)),
        enable_word_time_offsets=bool(config.get("enable_word_time_offsets", False)),
        max_alternatives=int(config.get("max_alternatives", 1)),
        model=config.get("model", "") or "",
        verbatim_transcripts=bool(config.get("verbatim_transcripts", False)),
        profanity_filter=bool(config.get("profanity_filter", False)),
        **({} if endpointing is None else {"endpointing_config": endpointing}),
    )


def _endpointing_config(endpointing: object) -> Message | None:
    try:
        fields: Final = _ENDPOINTING_FIELDS.validate_python(endpointing)
        return schema.EndpointingConfig(**fields) if fields else None
    except (ValueError, TypeError):
        # A missing, unknown or mistyped field falls back to Riva's default
        # endpointing rather than failing the whole request.
        return None


def _streaming_requests(config: Message, pcm_bytes: bytes) -> Iterator[Message]:
    yield schema.StreamingRecognizeRequest(
        streaming_config=schema.StreamingRecognitionConfig(config=config, interim_results=False)
    )
    for offset in range(0, len(pcm_bytes), _CHUNK_BYTES):
        yield schema.StreamingRecognizeRequest(audio_content=pcm_bytes[offset : offset + _CHUNK_BYTES])


def streaming_recognize(
    channel: grpc.Channel,
    config: Message,
    pcm_bytes: bytes,
    metadata: CallMetadata,
    timeout: float | None,
) -> Iterator[Message]:
    rpc: Final = channel.stream_stream(
        schema.STREAMING_RECOGNIZE_METHOD,
        request_serializer=schema.StreamingRecognizeRequest.SerializeToString,
        response_deserializer=schema.StreamingRecognizeResponse.FromString,
    )
    return rpc(_streaming_requests(config, pcm_bytes), metadata=metadata, timeout=timeout)


def recognize(
    channel: grpc.Channel,
    config: Message,
    pcm_bytes: bytes,
    metadata: CallMetadata,
    timeout: float | None,
) -> Sequence[Message]:
    rpc: Final = channel.unary_unary(
        schema.RECOGNIZE_METHOD,
        request_serializer=schema.RecognizeRequest.SerializeToString,
        response_deserializer=schema.RecognizeResponse.FromString,
    )
    pending: Final = rpc.future(schema.RecognizeRequest(config=config, audio=pcm_bytes), metadata=metadata)
    try:
        return (pending.result(timeout=timeout),)
    except Exception as e:
        if pending.done():
            raise
        pending.cancel()
        raise NvidiaRivaException(
            status_code=408,
            message=f"NVIDIA Riva offline recognition did not finish within {timeout} seconds",
        ) from e
