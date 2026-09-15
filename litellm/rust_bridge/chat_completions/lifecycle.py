from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import wraps
from inspect import signature
from typing import Final, ParamSpec, TypeVar, cast  # noqa: TID251  # native bindings are validated when loaded

from litellm.rust_bridge.bindings import BINDING_UNSET, BindingUnset
from litellm.rust_bridge.chat_completions.definition import COMPONENT
from litellm.rust_bridge.chat_completions.host import HOST
from litellm.rust_bridge.chat_completions.types import RustAchatCompletions, RustChatCompletions
from litellm.rust_bridge.configuration import CapabilityContext, DeliveryMode
from litellm.rust_bridge.runtime import ainvoke_lifecycle, invoke_lifecycle

Params = ParamSpec("Params")
ResultT = TypeVar("ResultT")


def _as_chat(value: object) -> RustChatCompletions | None:
    return cast(RustChatCompletions, value) if callable(value) else None  # cast-ok: callable checked at binding


def _as_achat(value: object) -> RustAchatCompletions | None:
    return cast(RustAchatCompletions, value) if callable(value) else None  # cast-ok: callable checked at binding


_CHAT: Final = COMPONENT.bind("chat_completions", validate=_as_chat)
_ACHAT: Final = COMPONENT.bind("achat_completions", validate=_as_achat)


def set_rust_chat_completions(
    *,
    chat_completions: RustChatCompletions | None | BindingUnset = BINDING_UNSET,
    achat_completions: RustAchatCompletions | None | BindingUnset = BINDING_UNSET,
) -> None:
    _CHAT.configure(chat_completions)
    _ACHAT.configure(achat_completions)


def _request(args: tuple[object, ...], kwargs: dict[str, object]) -> dict[str, object]:
    model: Final = args[0] if args else kwargs.get("model")
    messages: Final = args[1] if len(args) > 1 else kwargs.get("messages")
    stream: Final = kwargs.get("stream") is True
    return {  # mutable-ok: PyO3 requires an owned exact dict at admission
        "model": model,
        "messages": messages,
        **kwargs,
        "optional_params": {},  # mutable-ok: native projection fills parameters after admission
        "host_facts": {"stream": stream},  # mutable-ok: exact admission facts are passed by value
    }


def _context(request: dict[str, object]) -> CapabilityContext:
    model: Final = request.get("model")
    provider: Final = request.get("custom_llm_provider")
    return CapabilityContext(
        provider=provider if isinstance(provider, str) else "",
        model=model if isinstance(model, str) else "",
        delivery=DeliveryMode.STREAMING if request.get("stream") is True else DeliveryMode.COMPLETED,
    )


def wrap_sync(function: Callable[Params, ResultT]) -> Callable[Params, ResultT | object]:
    @wraps(function)
    def wrapped(
        *args: Params.args,
        **kwargs: Params.kwargs,  # kwargs-ok: preserves public SDK call shape
    ) -> ResultT | object:
        signature(function).bind(*args, **kwargs)
        call_args: Final = tuple(args)
        call_kwargs: Final = dict(kwargs)  # mutable-ok: PyO3 requires the original concrete kwargs dict
        request: Final = _request(call_args, call_kwargs)
        execution: Final = COMPONENT.resolve(_context(request))
        native: Final = execution.select(_CHAT)
        return invoke_lifecycle(
            execution=execution,
            native_call=(lambda: native(request, call_args, call_kwargs, HOST)) if native is not None else None,
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
        request: Final = _request(call_args, call_kwargs)
        execution: Final = COMPONENT.resolve(_context(request))
        native: Final = execution.select(_ACHAT)
        return await ainvoke_lifecycle(
            execution=execution,
            native_call=(lambda: native(request, call_args, call_kwargs, HOST)) if native is not None else None,
            python_fallback=lambda: function(*args, **kwargs),
        )

    return wrapped
