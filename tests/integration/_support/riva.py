from __future__ import annotations

import io
import threading
import wave
from array import array
from collections.abc import Iterable, Iterator
from concurrent import futures
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from queue import SimpleQueue
from typing import Final

import grpc
from riva.client.proto import riva_asr_pb2, riva_asr_pb2_grpc

SAMPLE_RATE_HZ: Final = 16000
OFFLINE_ONLY_REFUSAL: Final = (
    "Error: Unavailable model requested given these parameters: language_code=en-US; sample_rate=16000; type=online;"
)
_STALL_LIMIT_SECONDS: Final = 120


class Rpc(Enum):
    RECOGNIZE = "Recognize"
    STREAMING = "StreamingRecognize"


@dataclass(frozen=True, slots=True)
class Word:
    word: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class RivaCall:
    rpc: Rpc
    audio: bytes
    cancelled: bool


@dataclass(frozen=True, slots=True)
class RivaServer:
    address: str
    calls: SimpleQueue[RivaCall]

    def drain(self) -> tuple[RivaCall, ...]:
        return tuple(self.calls.get_nowait() for _ in range(self.calls.qsize()))


def _clip_samples(seconds: float) -> array[int]:
    return array("h", (((index * 37) % 2001 - 1000) * 16 for index in range(int(seconds * SAMPLE_RATE_HZ))))


def pcm_wav(seconds: float) -> bytes:
    buffer: Final = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE_HZ)
        writer.writeframes(_clip_samples(seconds).tobytes())
    return buffer.getvalue()


def pcm_bytes(seconds: float) -> int:
    return int(seconds * SAMPLE_RATE_HZ) * 2


def carries_clip(audio: bytes, seconds: float) -> bool:
    sent: Final = _clip_samples(seconds)
    if len(audio) != len(sent) * 2:
        return False
    received: Final = array("h", audio)
    return all(abs(left - right) <= 1 for left, right in zip(sent, received, strict=True))


class _Asr(riva_asr_pb2_grpc.RivaSpeechRecognitionServicer):
    def __init__(self, words: tuple[Word, ...], offline_only: bool, stall: bool, calls: SimpleQueue[RivaCall]) -> None:
        self._words: Final = words
        self._offline_only: Final = offline_only
        self._stall: Final = stall
        self._calls: Final = calls

    def _alternative(self, config: riva_asr_pb2.RecognitionConfig) -> riva_asr_pb2.SpeechRecognitionAlternative:
        timed: Final = self._words if config.enable_word_time_offsets else ()
        return riva_asr_pb2.SpeechRecognitionAlternative(
            transcript=" ".join(word.word for word in self._words),
            confidence=0.9,
            words=[
                riva_asr_pb2.WordInfo(word=word.word, start_time=word.start_ms, end_time=word.end_ms) for word in timed
            ],
        )

    def Recognize(
        self, request: riva_asr_pb2.RecognizeRequest, context: grpc.ServicerContext
    ) -> riva_asr_pb2.RecognizeResponse:
        if self._stall:
            released: Final = threading.Event()
            context.add_callback(released.set)
            released.wait(timeout=_STALL_LIMIT_SECONDS)
            self._calls.put(RivaCall(Rpc.RECOGNIZE, request.audio, cancelled=not context.is_active()))
            return riva_asr_pb2.RecognizeResponse()
        self._calls.put(RivaCall(Rpc.RECOGNIZE, request.audio, cancelled=False))
        return riva_asr_pb2.RecognizeResponse(
            results=[riva_asr_pb2.SpeechRecognitionResult(alternatives=[self._alternative(request.config)])]
        )

    def StreamingRecognize(
        self, request_iterator: Iterable[riva_asr_pb2.StreamingRecognizeRequest], context: grpc.ServicerContext
    ) -> Iterator[riva_asr_pb2.StreamingRecognizeResponse]:
        if self._offline_only:
            self._calls.put(RivaCall(Rpc.STREAMING, b"", cancelled=False))
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, OFFLINE_ONLY_REFUSAL)
        requests: Final = tuple(request_iterator)
        config: Final = requests[0].streaming_config.config if requests else riva_asr_pb2.RecognitionConfig()
        self._calls.put(
            RivaCall(Rpc.STREAMING, b"".join(request.audio_content for request in requests), cancelled=False)
        )
        yield riva_asr_pb2.StreamingRecognizeResponse(
            results=[riva_asr_pb2.StreamingRecognitionResult(alternatives=[self._alternative(config)], is_final=True)]
        )


@contextmanager
def riva_server(
    words: tuple[Word, ...],
    *,
    offline_only: bool = False,
    stall: bool = False,
    max_receive_bytes: int | None = None,
) -> Iterator[RivaServer]:
    calls: Final[SimpleQueue[RivaCall]] = SimpleQueue()
    options: Final = () if max_receive_bytes is None else (("grpc.max_receive_message_length", max_receive_bytes),)
    server: Final = grpc.server(futures.ThreadPoolExecutor(max_workers=8), options=options)
    riva_asr_pb2_grpc.add_RivaSpeechRecognitionServicer_to_server(_Asr(words, offline_only, stall, calls), server)
    port: Final = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield RivaServer(f"127.0.0.1:{port}", calls)
    finally:
        server.stop(grace=None).wait()
