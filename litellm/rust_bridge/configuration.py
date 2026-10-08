from __future__ import annotations

import os
from functools import lru_cache
from typing import Final

from pydantic import TypeAdapter, ValidationError

_GLOBAL_ENV_NAME: Final = "LITELLM_RUST"
_ENV_BOOL: Final = TypeAdapter(bool)


class _RustConfiguration:
    def __init__(self) -> None:
        self.override: bool | None = None


_CONFIGURATION: Final = _RustConfiguration()


@lru_cache(maxsize=16)
def _parse_env_bool(value: str | None) -> bool | None:
    """`LITELLM_RUST` as a bool; cached by raw value because `decide` runs per tokenizer call."""
    if value is None:
        return None
    try:
        return _ENV_BOOL.validate_python(value.strip())
    except ValidationError:
        return None


def rust_enabled() -> bool:
    """The global switch for optional native paths: `LITELLM_RUST` wins, then `litellm.rust(...)`, else off."""
    environment: Final = _parse_env_bool(os.getenv(_GLOBAL_ENV_NAME))
    if environment is not None:
        return environment
    override: Final = _CONFIGURATION.override
    return override if override is not None else False


def reset_rust_configuration() -> None:
    _CONFIGURATION.override = None


def rust(enabled: bool | None) -> None:
    """Set the process override for optional Rust paths.

    Routes the catalog marks required or unported ignore this switch, and an explicit
    ``LITELLM_RUST`` environment value wins over it.
    """
    _CONFIGURATION.override = enabled
