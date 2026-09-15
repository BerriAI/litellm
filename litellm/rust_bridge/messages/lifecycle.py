from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import wraps
from inspect import signature
from typing import Final, ParamSpec, TypeVar, cast  # noqa: TID251  # native bindings are validated when loaded

from litellm.rust_bridge.bindings import BINDING_UNSET, BindingUnset
from litellm.rust_bridge.messages.definition import COMPONENT
from litellm.rust_bridge.messages.host import HOST
from litellm.rust_bridge.messages.request import context, request
from litellm.rust_bridge.messages.types import RustAmessages, RustMessages
from litellm.rust_bridge.runtime import ainvoke_lifecycle, invoke_lifecycle

Params = ParamSpec("Params")
ResultT = TypeVar("ResultT")


def _as_messages(value: object) -> RustMessages | None:
    return cast(RustMessages, value) if callable(value) else None  # cast-ok: callable checked at binding


def _as_amessages(value: object) -> RustAmessages | None:
    return cast(RustAmessages, value) if callable(value) else None  # cast-ok: callable checked at binding


_MESSAGES: Final = COMPONENT.bind("messages", validate=_as_messages)
_AMESSAGES: Final = COMPONENT.bind("amessages", validate=_as_amessages)


def set_rust_messages(
    *,
    messages: RustMessages | None | BindingUnset = BINDING_UNSET,
    amessages: RustAmessages | None | BindingUnset = BINDING_UNSET,
) -> None:
    _MESSAGES.configure(messages)
    _AMESSAGES.configure(amessages)


def wrap_sync(function: Callable[Params, ResultT]) -> Callable[Params, ResultT | object]:
    @wraps(function)
    def wrapped(
        *args: Params.args,
        **kwargs: Params.kwargs,  # kwargs-ok: preserves public SDK call shape
    ) -> ResultT | object:
        signature(function).bind(*args, **kwargs)
        call_args: Final = tuple(args)
        call_kwargs: Final = dict(kwargs)  # mutable-ok: PyO3 requires the original concrete kwargs dict
        boundary_request: Final = request(call_args, call_kwargs)
        execution: Final = COMPONENT.resolve(context(boundary_request))
        native: Final = execution.select(_MESSAGES)
        return invoke_lifecycle(
            execution=execution,
            native_call=(lambda: native(boundary_request, call_args, call_kwargs, HOST))
            if native is not None
            else None,
            python_fallback=lambda: function(*args, **kwargs),
        )

    return wrapped


def wrap_async(function: Callable[Params, Awaitable[ResultT]]) -> Callable[Params, Awaitable[ResultT | object]]:
    @wraps(function)
    async def wrapped(
        *args: Params.args,
        **kwargs: Params.kwargs,  # kwargs-ok: preserves public SDK call shape
    ) -> ResultT | object:
        signature(function).bind(*args, **kwargs)
        call_args: Final = tuple(args)
        call_kwargs: Final = dict(kwargs)  # mutable-ok: PyO3 requires the original concrete kwargs dict
        boundary_request: Final = request(call_args, call_kwargs)
        execution: Final = COMPONENT.resolve(context(boundary_request))
        native: Final = execution.select(_AMESSAGES)
        return await ainvoke_lifecycle(
            execution=execution,
            native_call=(lambda: native(boundary_request, call_args, call_kwargs, HOST))
            if native is not None
            else None,
            python_fallback=lambda: function(*args, **kwargs),
        )

    return wrapped
