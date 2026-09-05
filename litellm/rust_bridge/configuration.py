from __future__ import annotations

import os
import warnings
from typing import Final

DEFAULT_RUST_ENABLED: Final = False
_TRUE_ENV_VALUES: Final = frozenset({"1", "true", "yes", "on"})
_GLOBAL_ENV_NAME: Final = "LITELLM_RUST"
_LEGACY_OCR_ENV_NAME: Final = "LITELLM_USE_RUST_OCR"


class _RustConfiguration:
    def __init__(self) -> None:
        self.override: bool | None = None


_CONFIGURATION: Final = _RustConfiguration()


def _parse_env_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.strip().lower() in _TRUE_ENV_VALUES


def resolve_rust_enabled(
    *,
    request_override: bool | None,
    process_override: bool | None,
    environment_override: bool | None,
    legacy_environment_override: bool | None = None,
    release_default: bool = DEFAULT_RUST_ENABLED,
) -> bool:
    if request_override is not None:
        return request_override
    if process_override is not None:
        return process_override
    if environment_override is not None:
        return environment_override
    if legacy_environment_override is not None:
        return legacy_environment_override
    return release_default


def rust_enabled(*, request_override: bool | None = None) -> bool:
    if request_override is not None:
        return request_override
    process_override: Final = _CONFIGURATION.override
    if process_override is not None:
        return process_override
    global_override: Final = _parse_env_bool(os.getenv(_GLOBAL_ENV_NAME))
    legacy_override: Final = None if global_override is not None else _parse_env_bool(os.getenv(_LEGACY_OCR_ENV_NAME))
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
        legacy_environment_override=legacy_override,
    )


def rust_ocr_enabled(*, request_override: bool | None = None) -> bool:
    return rust_enabled(request_override=request_override)


def reset_rust_configuration() -> None:
    _CONFIGURATION.override = None


def rust(enabled: bool) -> None:
    """Set the process override for optional Rust paths.

    Rust-only paths, including Bedrock transcription, are not controlled by this switch.
    """
    _CONFIGURATION.override = enabled
