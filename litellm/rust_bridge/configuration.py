from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from litellm.rust_bridge.messages import RustAmessages, RustMessages
    from litellm.rust_bridge.ocr import RustAocr, RustOcr
    from litellm.rust_bridge.responses_websocket import RustResponsesWebSocketConnection
    from litellm.rust_bridge.transcription import RustAtranscription, RustTranscription

DEFAULT_RUST_ENABLED: Final = False
_TRUE_ENV_VALUES: Final = frozenset({"1", "true", "yes", "on"})
_FALSE_ENV_VALUES: Final = frozenset({"0", "false", "no", "off"})
_GLOBAL_ENV_NAME: Final = "LITELLM_RUST"
_LEGACY_OCR_ENV_NAME: Final = "LITELLM_USE_RUST_OCR"


class _Unset:
    pass


_UNSET: Final = _Unset()


class _RustConfiguration:
    def __init__(self) -> None:
        self.override: bool | None = None


_CONFIGURATION: Final = _RustConfiguration()


def _parse_env_bool(name: str, value: str | None) -> bool | None:
    if value is None:
        return None
    normalized: Final = value.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    accepted: Final = ", ".join(sorted(_TRUE_ENV_VALUES | _FALSE_ENV_VALUES))
    raise ValueError(f"{name} must be one of: {accepted}")


def resolve_rust_enabled(
    *,
    request_override: bool | None,
    process_override: bool | None,
    environment_override: bool | None,
    legacy_ocr_override: bool | None = None,
    release_default: bool = DEFAULT_RUST_ENABLED,
) -> bool:
    if request_override is not None:
        return request_override
    if process_override is not None:
        return process_override
    if environment_override is not None:
        return environment_override
    if legacy_ocr_override is not None:
        return legacy_ocr_override
    return release_default


def rust_enabled(*, request_override: bool | None = None) -> bool:
    if request_override is not None:
        return request_override
    process_override: Final = _CONFIGURATION.override
    if process_override is not None:
        return process_override
    return resolve_rust_enabled(
        request_override=None,
        process_override=None,
        environment_override=_parse_env_bool(_GLOBAL_ENV_NAME, os.getenv(_GLOBAL_ENV_NAME)),
    )


def rust_ocr_enabled(*, request_override: bool | None = None) -> bool:
    if request_override is not None:
        return request_override
    process_override: Final = _CONFIGURATION.override
    if process_override is not None:
        return process_override
    global_override: Final = _parse_env_bool(_GLOBAL_ENV_NAME, os.getenv(_GLOBAL_ENV_NAME))
    legacy_override: Final = (
        None if global_override is not None else _parse_env_bool(_LEGACY_OCR_ENV_NAME, os.getenv(_LEGACY_OCR_ENV_NAME))
    )
    if legacy_override is not None:
        warnings.warn(
            f"{_LEGACY_OCR_ENV_NAME} is deprecated; use {_GLOBAL_ENV_NAME} instead",
            DeprecationWarning,
            stacklevel=2,
        )
    return resolve_rust_enabled(
        request_override=None,
        process_override=None,
        environment_override=global_override,
        legacy_ocr_override=legacy_override,
    )


def reset_rust_configuration() -> None:
    _CONFIGURATION.override = None


def use_litellm_rust(
    enabled: bool = True,
    *,
    ocr: RustOcr | None | _Unset = _UNSET,
    aocr: RustAocr | None | _Unset = _UNSET,
    messages: RustMessages | None | _Unset = _UNSET,
    amessages: RustAmessages | None | _Unset = _UNSET,
    responses_websocket: type[RustResponsesWebSocketConnection] | None | _Unset = _UNSET,
    transcription: RustTranscription | None | _Unset = _UNSET,
    atranscription: RustAtranscription | None | _Unset = _UNSET,
) -> None:
    """Set the process override for optional Rust paths.

    Rust-only paths, including Bedrock transcription, are not controlled by this switch.
    """
    _CONFIGURATION.override = enabled
    bindings: Final = (ocr, aocr, messages, amessages, responses_websocket, transcription, atranscription)
    if all(isinstance(binding, _Unset) for binding in bindings):
        return
    warnings.warn(
        "Injecting Rust bridge implementations through use_litellm_rust() is deprecated; "
        "use the internal bridge setters in tests",
        DeprecationWarning,
        stacklevel=2,
    )

    if not isinstance(ocr, _Unset) or not isinstance(aocr, _Unset):
        from litellm.rust_bridge.ocr import set_rust_ocr

        if not isinstance(ocr, _Unset):
            set_rust_ocr(ocr=ocr)
        if not isinstance(aocr, _Unset):
            set_rust_ocr(aocr=aocr)
    if not isinstance(messages, _Unset) or not isinstance(amessages, _Unset):
        from litellm.rust_bridge.messages import set_rust_messages

        if not isinstance(messages, _Unset):
            set_rust_messages(messages=messages)
        if not isinstance(amessages, _Unset):
            set_rust_messages(amessages=amessages)
    if not isinstance(responses_websocket, _Unset):
        from litellm.rust_bridge.responses_websocket import set_rust_responses_websocket

        set_rust_responses_websocket(connection=responses_websocket)
    if not isinstance(transcription, _Unset) or not isinstance(atranscription, _Unset):
        from litellm.rust_bridge.transcription import configure_rust_transcription

        if not isinstance(transcription, _Unset):
            configure_rust_transcription(transcription=transcription)
        if not isinstance(atranscription, _Unset):
            configure_rust_transcription(atranscription=atranscription)
