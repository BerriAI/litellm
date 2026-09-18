from __future__ import annotations

import os
from enum import Enum, auto
from typing import Final

from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

_GLOBAL_ENV_NAME: Final = "LITELLM_RUST"
_ENV_BOOL: Final = TypeAdapter(bool)


class Rollout(Enum):
    PYTHON_ONLY = auto()
    RUST_OPT_IN = auto()
    RUST_OPT_OUT = auto()
    RUST_REQUIRED = auto()


class Decision(Enum):
    PYTHON = auto()
    RUST_WITH_FALLBACK = auto()
    RUST_REQUIRED = auto()


class _RustConfiguration:
    def __init__(self) -> None:
        self.override: bool | None = None


_CONFIGURATION: Final = _RustConfiguration()


def _parse_env_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    try:
        return _ENV_BOOL.validate_python(value.strip())
    except ValidationError:
        return None


def decide(
    rollout: Rollout,
    *,
    process_override: bool | None,
    environment_override: bool | None,
) -> Decision:
    match rollout:
        case Rollout.PYTHON_ONLY:
            return Decision.PYTHON
        case Rollout.RUST_REQUIRED:
            return Decision.RUST_REQUIRED
        case Rollout.RUST_OPT_IN | Rollout.RUST_OPT_OUT:
            switch: Final = (
                environment_override
                if environment_override is not None
                else process_override
                if process_override is not None
                else rollout is Rollout.RUST_OPT_OUT
            )
            return Decision.RUST_WITH_FALLBACK if switch else Decision.PYTHON
        case _:
            assert_never(rollout)


def decision(rollout: Rollout) -> Decision:
    return decide(
        rollout,
        process_override=_CONFIGURATION.override,
        environment_override=_parse_env_bool(os.getenv(_GLOBAL_ENV_NAME)),
    )


def rust_enabled() -> bool:
    return decision(Rollout.RUST_OPT_IN) is not Decision.PYTHON


def reset_rust_configuration() -> None:
    _CONFIGURATION.override = None


def rust(enabled: bool | None) -> None:
    """Set the process override for optional Rust paths.

    ``PYTHON_ONLY`` and ``RUST_REQUIRED`` routes in the catalog ignore this switch,
    and an explicit ``LITELLM_RUST`` environment value wins over it.
    """
    _CONFIGURATION.override = enabled
