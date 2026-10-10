from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Final, Protocol, TypeVar, cast

from litellm.rust_bridge.bindings import NativeBinding

ResultT: Final = TypeVar("ResultT")


class NativeDiagnosticProcessor(Protocol):
    def redact_text(self, text: str) -> str: ...
    def redact_structured_text(self, key: str | None, text: str) -> str: ...
    def redact_client_message(self, text: str) -> str: ...
    def process_diagnostic(
        self,
        message: str,
        exception: str | None,
        stack: str | None,
        leaves: tuple[tuple[str | None, str], ...],
        policy: tuple[bool, int, int],
    ) -> tuple[str, str | None, str | None, Sequence[str], bool]: ...
    def scrub_access_arguments(self, arguments: tuple[str, ...]) -> Sequence[str]: ...


class NativeDiagnosticFactory(Protocol):
    def __call__(self, minimum_custom_key_length: int) -> NativeDiagnosticProcessor: ...


def _as_factory(value: object) -> NativeDiagnosticFactory | None:
    if not isinstance(value, type):
        return None
    return cast(NativeDiagnosticFactory, value)  # cast-ok: PyO3 factory must be a type


PROCESSOR: Final = NativeBinding("NativeDiagnosticProcessor", validate=_as_factory)


@lru_cache(maxsize=4)
def _construct(factory: NativeDiagnosticFactory, minimum_custom_key_length: int) -> NativeDiagnosticProcessor:
    return factory(minimum_custom_key_length)


def run(native: Callable[[NativeDiagnosticProcessor], ResultT], python: Callable[[], ResultT]) -> ResultT:
    from litellm.constants import MINIMUM_CUSTOM_KEY_LENGTH
    from litellm.rust_bridge.catalog import LoggerContext, decision
    from litellm.rust_bridge.configuration import Decision

    selected: Final = decision(LoggerContext())
    if selected is Decision.PYTHON:
        return python()
    factory: Final = PROCESSOR.load()
    if factory is None:
        return python()
    try:
        return native(_construct(factory, MINIMUM_CUSTOM_KEY_LENGTH))
    except Exception:
        return python()
