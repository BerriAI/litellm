from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override

from .._utils import LazyProxy
from ._common import MissingDependencyError, format_instructions

if TYPE_CHECKING:
    import sounddevice as sounddevice  # type: ignore


SOUNDDEVICE_INSTRUCTIONS = format_instructions(
    library="sounddevice", extra="voice_helpers"
)


class SounddeviceProxy(LazyProxy[Any]):
    @override
    def __load__(self) -> Any:
        try:
            import sounddevice  # type: ignore
        except ImportError as err:
            raise MissingDependencyError(SOUNDDEVICE_INSTRUCTIONS) from err

        return sounddevice


if not TYPE_CHECKING:
    sounddevice = SounddeviceProxy()
