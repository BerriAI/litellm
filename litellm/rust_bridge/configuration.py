from __future__ import annotations

import os
from enum import Enum, auto
from typing import Final

_TRUE_ENV_VALUES: Final = frozenset({"1", "true", "yes", "on"})
_GLOBAL_ENV_NAME: Final = "LITELLM_RUST"


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
    return value.strip().lower() in _TRUE_ENV_VALUES


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
                process_override
                if process_override is not None
                else environment_override
                if environment_override is not None
                else rollout is Rollout.RUST_OPT_OUT
            )
            return Decision.RUST_WITH_FALLBACK if switch else Decision.PYTHON


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

    ``PYTHON_ONLY`` and ``RUST_REQUIRED`` routes in the catalog ignore this switch.
    """
    _CONFIGURATION.override = enabled
