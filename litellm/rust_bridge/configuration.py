from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

DEFAULT_RUST_ENABLED: Final = False
_GLOBAL_ENV_NAME: Final = "LITELLM_RUST"
_ENV_BOOL: Final = TypeAdapter(bool)


class RouteName(str, Enum):
    OCR = "ocr"
    MESSAGES = "messages"
    CHAT_COMPLETIONS = "chat_completions"
    TRANSCRIPTION = "transcription"
    EMBEDDINGS = "embeddings"
    RERANK = "rerank"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDIT = "image_edit"
    SPEECH = "speech"
    MODERATION = "moderation"
    RESPONSES = "responses"


class RouteMode(str, Enum):
    OPTIONAL = "optional"
    REQUIRED = "required"


@dataclass(frozen=True, slots=True)
class RoutePolicy:
    default_enabled: bool = False
    mode: RouteMode = RouteMode.OPTIONAL
    environment_opt_out: bool = False


ROUTE_POLICIES: Final = MappingProxyType(
    {
        RouteName.OCR: RoutePolicy(default_enabled=True, environment_opt_out=True),
        RouteName.MESSAGES: RoutePolicy(),
        RouteName.CHAT_COMPLETIONS: RoutePolicy(),
        RouteName.TRANSCRIPTION: RoutePolicy(mode=RouteMode.REQUIRED),
        RouteName.EMBEDDINGS: RoutePolicy(),
        RouteName.RERANK: RoutePolicy(),
        RouteName.IMAGE_GENERATION: RoutePolicy(),
        RouteName.IMAGE_EDIT: RoutePolicy(),
        RouteName.SPEECH: RoutePolicy(),
        RouteName.MODERATION: RoutePolicy(),
        RouteName.RESPONSES: RoutePolicy(),
    }
)


class _RustConfiguration:
    def __init__(self) -> None:
        self.override: bool | None = None


_CONFIGURATION: Final = _RustConfiguration()


def _parse_env_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return _ENV_BOOL.validate_python(value.strip())


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


def rust_enabled(route: RouteName | None = None) -> bool:
    policy: Final = ROUTE_POLICIES[route] if route is not None else RoutePolicy()
    if policy.mode is RouteMode.REQUIRED:
        return True
    environment: Final = _parse_env_bool(os.getenv(_GLOBAL_ENV_NAME))
    if policy.environment_opt_out and environment is False:
        return False
    return resolve_rust_enabled(
        process_override=_CONFIGURATION.override,
        environment_override=environment,
        release_default=policy.default_enabled,
    )


def rust_ocr_enabled() -> bool:
    return rust_enabled(RouteName.OCR)


def reset_rust_configuration() -> None:
    _CONFIGURATION.override = None


def rust(enabled: bool) -> None:
    """Set the process override for optional Rust paths.

    Rust-only paths, including Bedrock transcription, are not controlled by this switch.
    """
    _CONFIGURATION.override = enabled
