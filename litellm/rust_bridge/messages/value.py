"""Native Messages bindings."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import (
    Final,
    TypeVar,
    cast,  # noqa: TID251  # native callable signatures are checked by bridge contract tests
)

import httpx

from litellm.rust_bridge.bindings import BINDING_UNSET, BindingUnset
from litellm.rust_bridge.configuration import CapabilityContext, DeliveryMode
from litellm.rust_bridge.messages.definition import COMPONENT
from litellm.rust_bridge.messages.types import RustAmessages, RustMessages
from litellm.rust_bridge.runtime import BridgeErrorContext, ainvoke, invoke
from litellm.rust_bridge.timeouts import timeout_to_seconds


def _as_messages(value: object) -> RustMessages | None:
    return cast(RustMessages, value) if callable(value) else None  # cast-ok: validated callable native binding


def _as_amessages(value: object) -> RustAmessages | None:
    return cast(RustAmessages, value) if callable(value) else None  # cast-ok: validated callable native binding


_MESSAGES: Final = COMPONENT.bind("messages", validate=_as_messages)
_AMESSAGES: Final = COMPONENT.bind("amessages", validate=_as_amessages)
ResultT = TypeVar("ResultT")


def set_rust_messages(
    *,
    messages: RustMessages | None | BindingUnset = BINDING_UNSET,
    amessages: RustAmessages | None | BindingUnset = BINDING_UNSET,
) -> None:
    _MESSAGES.configure(messages)
    _AMESSAGES.configure(amessages)


def load_rust_messages() -> RustMessages | None:
    return COMPONENT.resolve().select(_MESSAGES)


def load_rust_amessages() -> RustAmessages | None:
    return COMPONENT.resolve().select(_AMESSAGES)


def messages(
    *,
    model: str,
    body: Mapping[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, object] | None,
    timeout: float | httpx.Timeout | None,
    has_agentic_hook: bool = False,
    on_request: Callable[[], None] = lambda: None,
    python_fallback: Callable[[], ResultT],
    adapt: Callable[[dict[str, object]], ResultT],
) -> ResultT:
    execution: Final = COMPONENT.resolve(
        CapabilityContext(
            provider=custom_llm_provider or "",
            model=model,
            delivery=DeliveryMode.STREAMING if body.get("stream") is True else DeliveryMode.COMPLETED,
        )
    )
    rust_messages: Final = execution.select(_MESSAGES)
    native_call: Final[Callable[[], dict[str, object]] | None] = (
        (
            lambda: rust_messages(
                model=model,
                body=body,
                has_agentic_hook=has_agentic_hook,
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout),
                on_request=on_request,
            )
        )
        if rust_messages is not None
        else None
    )
    return invoke(
        execution=execution,
        native_call=native_call,
        python_fallback=python_fallback,
        adapt=adapt,
        context=BridgeErrorContext(
            route=COMPONENT.name.value,
            provider=custom_llm_provider or "",
            model=model,
        ),
    )


async def amessages(
    *,
    model: str,
    body: Mapping[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, object] | None,
    timeout: float | httpx.Timeout | None,
    has_agentic_hook: bool = False,
    on_request: Callable[[], None] = lambda: None,
    python_fallback: Callable[[], Awaitable[ResultT]],
    adapt: Callable[[dict[str, object]], Awaitable[ResultT]],
) -> ResultT:
    execution: Final = COMPONENT.resolve(
        CapabilityContext(
            provider=custom_llm_provider or "",
            model=model,
            delivery=DeliveryMode.STREAMING if body.get("stream") is True else DeliveryMode.COMPLETED,
        )
    )
    rust_amessages: Final = execution.select(_AMESSAGES)
    native_call: Final[Callable[[], Awaitable[dict[str, object]]] | None] = (
        (
            lambda: rust_amessages(
                model=model,
                body=body,
                has_agentic_hook=has_agentic_hook,
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout),
                on_request=on_request,
            )
        )
        if rust_amessages is not None
        else None
    )

    return await ainvoke(
        execution=execution,
        native_call=native_call,
        python_fallback=python_fallback,
        adapt=adapt,
        context=BridgeErrorContext(
            route=COMPONENT.name.value,
            provider=custom_llm_provider or "",
            model=model,
        ),
    )
