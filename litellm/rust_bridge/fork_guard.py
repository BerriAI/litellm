"""Fork safety of the Rust extension.

Its runtime threads do not survive ``fork``, so a child forked after the first native call
cannot run native routes: it raises ``ForkedAfterNativeRuntimeStarted`` instead of hanging.
Fork before the first native call, or start workers with ``spawn`` / ``forkserver``.

A process whose job is to fork workers (the gunicorn master under ``preload``) reserves itself:
from then on any native route called in it raises ``ProcessReservedForForking`` at the call
site, so the runtime can never start there. Workers forked from it are unaffected.
"""

from __future__ import annotations

from typing import Final

from litellm.rust_bridge.loader import get_native_bridge


class NativeStateStartedBeforeFork(RuntimeError):
    pass


class _NeverRaised(RuntimeError):
    """Stands in for a native exception when the extension is unavailable or predates it."""


_native: Final = get_native_bridge()
ForkedAfterNativeRuntimeStarted: Final[type[RuntimeError]] = getattr(
    _native, "ForkedAfterNativeRuntimeStarted", _NeverRaised
)
ProcessReservedForForking: Final[type[RuntimeError]] = getattr(_native, "ProcessReservedForForking", _NeverRaised)


def reserve_process_for_forking(where: str) -> None:
    """Forbid native routes in this process. Raises if one already ran here."""
    native: Final = get_native_bridge()
    reserve: Final = getattr(native, "reserve_process_for_forking", None)
    if not callable(reserve):
        return
    try:
        reserve()
    except RuntimeError as error:
        raise NativeStateStartedBeforeFork(
            f"The LiteLLM Rust extension already ran a native route in {where}, and its runtime "
            "threads do not survive fork(). Move the native call (warm-up, health check, "
            "import-time initialization) into the worker, after the fork."
        ) from error
