# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ExtensionHostSettings:
    endpoint: str
    token: str
    connect_timeout_seconds: float = 5.0
    hook_timeout_seconds: float = 30.0
    callback_queue_size: int = 1_000
    callback_batch_size: int = 50
    gateway_listen: str | None = None


def settings_from_config(
    config: dict[str, object],  # mutable-ok: LiteLLM compatibility payload
) -> ExtensionHostSettings | None:
    general_settings: Final = config.get("general_settings")
    if not isinstance(general_settings, dict):
        return None
    raw: Final = general_settings.get("python_extension_host")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("general_settings.python_extension_host must be an object")
    endpoint: Final = raw.get("endpoint")
    token: Final = raw.get("token")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("python_extension_host.endpoint is required")
    if not isinstance(token, str) or not token:
        raise ValueError("python_extension_host.token is required")
    resolved_token: Final = _resolve_env(token)
    return ExtensionHostSettings(
        endpoint=endpoint,
        token=resolved_token,
        connect_timeout_seconds=float(raw.get("connect_timeout_seconds", 5)),
        hook_timeout_seconds=float(raw.get("hook_timeout_seconds", 30)),
        callback_queue_size=int(raw.get("callback_queue_size", 1_000)),
        callback_batch_size=int(raw.get("callback_batch_size", 50)),
        gateway_listen=(str(raw["gateway_listen"]) if raw.get("gateway_listen") else None),
    )


def _resolve_env(value: str) -> str:
    if not value.startswith("os.environ/"):
        return value
    name: Final = value.removeprefix("os.environ/")
    resolved: Final = os.environ.get(name)
    if not resolved:
        raise ValueError(f"environment variable {name!r} is required for python_extension_host.token")
    return resolved
