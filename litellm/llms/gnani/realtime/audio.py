import asyncio
import audioop
import sys
from typing import Final

from litellm.llms.base_llm.realtime.transcription_protocol import RealtimeTranscriptionProtocolError

WIRE_RATE: Final = 16_000
_FRAME_BYTES: Final = 1024
_FRAME_SECONDS: Final = _FRAME_BYTES / (2 * WIRE_RATE)
_INPUT_RATES: Final = frozenset((8_000, 16_000, 24_000, 44_100, 48_000))
_MAX_APPEND_SECONDS: Final = 4


class GnaniAudioStream:
    def __init__(self) -> None:
        self.rate: int = 24_000
        self._remainder: bytes = b""
        self._state: tuple[int, tuple[tuple[int, int], ...]] | None = None
        self._sent_audio: bool = False
        self._next_frame: float = 0

    def set_rate(self, rate: int) -> None:
        if rate not in _INPUT_RATES:
            raise RealtimeTranscriptionProtocolError("Unsupported Gnani PCM sample rate")
        if self._sent_audio and rate != self.rate:
            raise RealtimeTranscriptionProtocolError("Open a new session to change the audio rate after sending audio")
        self.rate = rate

    @property
    def max_encoded_append_bytes(self) -> int:
        return 4 * ((self.rate * 2 * _MAX_APPEND_SECONDS + 2) // 3)

    def append(self, pcm: bytes) -> tuple[bytes, ...]:
        if not pcm:
            return ()
        self._sent_audio = True
        native: Final = pcm if sys.byteorder == "little" else audioop.byteswap(pcm, 2)
        converted, self._state = audioop.ratecv(native, 2, 1, self.rate, WIRE_RATE, self._state)
        frames: Final = self._remainder + (converted if sys.byteorder == "little" else audioop.byteswap(converted, 2))
        end: Final = len(frames) - len(frames) % _FRAME_BYTES
        self._remainder = frames[end:]
        return tuple(frames[offset : offset + _FRAME_BYTES] for offset in range(0, end, _FRAME_BYTES))

    def commit(self) -> tuple[str | bytes, ...]:
        if self._state is None:
            raise RealtimeTranscriptionProtocolError("Cannot commit an empty buffer")
        last: Final = self._remainder
        self._remainder = b""
        self._state = None
        return (last.ljust(_FRAME_BYTES, b"\0"), '{"type":"commit"}') if last else ('{"type":"commit"}',)

    async def pace(self) -> None:
        loop: Final = asyncio.get_running_loop()
        await asyncio.sleep(max(0.0, self._next_frame - loop.time()))
        self._next_frame = loop.time() + _FRAME_SECONDS
