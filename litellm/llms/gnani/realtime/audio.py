import asyncio
import audioop
import math
import sys
import time
from collections.abc import Awaitable, Callable
from functools import reduce
from types import MappingProxyType
from typing import Final

from litellm.llms.base_llm.realtime.transcription_protocol import RealtimeTranscriptionProtocolError

WIRE_RATE: Final = 16_000
_FRAME_BYTES: Final = 1024
_FRAME_SECONDS: Final = _FRAME_BYTES / (2 * WIRE_RATE)
_INPUT_RATES: Final = frozenset((8_000, 16_000, 24_000, 44_100, 48_000))
_MAX_APPEND_SECONDS: Final = 4
_ANTI_ALIAS_CUTOFF_HZ: Final = 8_000
_ANTI_ALIAS_TRANSITION_HZ: Final = 2_000
_HAMMING_TRANSITION_WIDTH: Final = 3.3


def _sinc(x: float) -> float:
    return 1.0 if x == 0 else math.sin(math.pi * x) / (math.pi * x)


def _hamming(index: int, count: int) -> float:
    return 0.54 - 0.46 * math.cos(2 * math.pi * index / (count - 1))


def _anti_alias_taps(rate: int) -> tuple[float, ...]:
    if rate <= WIRE_RATE:
        return ()
    count: Final = int(_HAMMING_TRANSITION_WIDTH * rate / _ANTI_ALIAS_TRANSITION_HZ) | 1
    middle: Final = count // 2
    taps: Final = tuple(
        _sinc(2 * _ANTI_ALIAS_CUTOFF_HZ / rate * (index - middle)) * _hamming(index, count) for index in range(count)
    )
    gain: Final = sum(taps)
    return tuple(tap / gain for tap in taps)


_ANTI_ALIAS_TAPS: Final = MappingProxyType({rate: _anti_alias_taps(rate) for rate in _INPUT_RATES})


class GnaniAudioStream:
    def __init__(
        self,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._monotonic: Final = monotonic
        self._sleep: Final = sleep
        self.rate: int = 24_000
        self._taps: tuple[float, ...] = _ANTI_ALIAS_TAPS[self.rate]
        self._history: bytes = self._silent_history()
        self._remainder: bytes = b""
        self._state: tuple[int, tuple[tuple[int, int], ...]] | None = None
        self._sent_audio: bool = False
        self._next_frame: float = 0

    def set_rate(self, rate: int) -> None:
        if rate not in _INPUT_RATES:
            raise RealtimeTranscriptionProtocolError("Unsupported Gnani PCM sample rate")
        if rate == self.rate:
            return
        if self._sent_audio:
            raise RealtimeTranscriptionProtocolError("Open a new session to change the audio rate after sending audio")
        self.rate = rate
        self._taps = _ANTI_ALIAS_TAPS[rate]
        self._history = self._silent_history()

    @property
    def max_encoded_append_bytes(self) -> int:
        return 4 * ((self.rate * 2 * _MAX_APPEND_SECONDS + 2) // 3)

    def _silent_history(self) -> bytes:
        return bytes(4 * max(len(self._taps) - 1, 0))

    def _anti_aliased(self, pcm: bytes) -> bytes:
        if not self._taps:
            return pcm
        wide: Final = self._history + audioop.lin2lin(pcm, 2, 4)
        length: Final = 2 * len(pcm)
        self._history = wide[length:]
        terms: Final = (
            audioop.mul(wide[4 * index : 4 * index + length], 4, tap / 2) for index, tap in enumerate(self._taps)
        )
        return audioop.mul(audioop.lin2lin(reduce(lambda total, term: audioop.add(total, term, 4), terms), 4, 2), 2, 2)

    def append(self, pcm: bytes) -> tuple[bytes, ...]:
        if not pcm:
            return ()
        self._sent_audio = True
        native: Final = pcm if sys.byteorder == "little" else audioop.byteswap(pcm, 2)
        converted, self._state = audioop.ratecv(self._anti_aliased(native), 2, 1, self.rate, WIRE_RATE, self._state)
        frames: Final = self._remainder + (converted if sys.byteorder == "little" else audioop.byteswap(converted, 2))
        end: Final = len(frames) - len(frames) % _FRAME_BYTES
        self._remainder = frames[end:]
        return tuple(frames[offset : offset + _FRAME_BYTES] for offset in range(0, end, _FRAME_BYTES))

    def commit(self) -> tuple[str | bytes, ...]:
        if self._state is None:
            raise RealtimeTranscriptionProtocolError("Cannot commit an empty buffer")
        filter_tail: Final = self.append(bytes(2 * (len(self._taps) // 2)))
        last: Final = self._remainder
        self._remainder = b""
        self._state = None
        self._history = self._silent_history()
        padded: Final = (last.ljust(_FRAME_BYTES, b"\0"),) if last else ()
        return (*filter_tail, *padded, '{"type":"commit"}')

    async def pace(self) -> None:
        now: Final = self._monotonic()
        due: Final = max(now, self._next_frame)
        await self._sleep(due - now)
        self._next_frame = due + _FRAME_SECONDS
