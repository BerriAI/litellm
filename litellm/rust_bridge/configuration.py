from __future__ import annotations

import os
from typing import Final

DEFAULT_RUST_ENABLED: Final = False
_TRUE_ENV_VALUES: Final = frozenset({"1", "true", "yes", "on"})
_GLOBAL_ENV_NAME: Final = "LITELLM_RUST"


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
    process_override: bool | None,
    environment_override: bool | None,
    release_default: bool = DEFAULT_RUST_ENABLED,
) -> bool:
    if process_override is not None:
        return process_override
    if environment_override is not None:
        return environment_override
    return release_default


def rust_enabled() -> bool:
    return resolve_rust_enabled(
        process_override=_CONFIGURATION.override,
        environment_override=_parse_env_bool(os.getenv(_GLOBAL_ENV_NAME)),
    )


def reset_rust_configuration() -> None:
    _CONFIGURATION.override = None


def rust(enabled: bool) -> None:
    """Set the process override for optional Rust paths.

    Rust-only paths, including Bedrock transcription, are not controlled by this switch.
    """
    _CONFIGURATION.override = enabled
