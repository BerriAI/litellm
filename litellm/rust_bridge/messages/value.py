"""Native Messages bindings."""

from __future__ import annotations

from typing import (
    Final,
    cast,  # noqa: TID251  # native callable signatures are checked by bridge contract tests
)

import httpx

from litellm.rust_bridge.bindings import BINDING_UNSET, BindingUnset
from litellm.rust_bridge.configuration import RouteName
from litellm.rust_bridge.messages.types import RustAmessages, RustMessages
from litellm.rust_bridge.route import NativeRoute
from litellm.rust_bridge.timeouts import timeout_to_seconds

ROUTE: Final = NativeRoute(RouteName.MESSAGES)


def _as_messages(value: object) -> RustMessages | None:
    return cast(RustMessages, value) if callable(value) else None  # cast-ok: validated callable native binding


def _as_amessages(value: object) -> RustAmessages | None:
    return cast(RustAmessages, value) if callable(value) else None  # cast-ok: validated callable native binding


_MESSAGES: Final = ROUTE.bind("messages", validate=_as_messages)
_AMESSAGES: Final = ROUTE.bind("amessages", validate=_as_amessages)


def set_rust_messages(
    *,
    messages: RustMessages | None | BindingUnset = BINDING_UNSET,
    amessages: RustAmessages | None | BindingUnset = BINDING_UNSET,
) -> None:
    _MESSAGES.configure(messages)
    _AMESSAGES.configure(amessages)


def load_rust_messages() -> RustMessages | None:
    return ROUTE.select(_MESSAGES)


def load_rust_amessages() -> RustAmessages | None:
    return ROUTE.select(_AMESSAGES)


def messages(
    *,
    model: str,
    body: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    timeout: float | httpx.Timeout | None,
) -> dict[str, object] | None:
    rust_messages: Final = load_rust_messages()
    if rust_messages is None:
        return None
    return rust_messages(
        model=model,
        body=body,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        timeout_seconds=timeout_to_seconds(timeout),
    )


async def amessages(
    *,
    model: str,
    body: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    timeout: float | httpx.Timeout | None,
) -> dict[str, object] | None:
    rust_amessages: Final = load_rust_amessages()
    if rust_amessages is None:
        return None
    return await rust_amessages(
        model=model,
        body=body,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        timeout_seconds=timeout_to_seconds(timeout),
    )
