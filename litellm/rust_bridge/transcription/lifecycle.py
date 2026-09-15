from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import wraps
from inspect import signature
from typing import Final, ParamSpec, TypeVar, cast  # noqa: TID251  # native bindings are validated when loaded

from litellm.rust_bridge.bindings import BINDING_UNSET, BindingUnset
from litellm.rust_bridge.configuration import ExecutionDecision
from litellm.rust_bridge.runtime import ainvoke_lifecycle, invoke_lifecycle
from litellm.rust_bridge.transcription.definition import COMPONENT
from litellm.rust_bridge.transcription.host import HOST
from litellm.rust_bridge.transcription.request import context, request
from litellm.rust_bridge.transcription.types import RustAtranscription, RustTranscription

Params = ParamSpec("Params")
ResultT = TypeVar("ResultT")


def _as_transcription(value: object) -> RustTranscription | None:
    return cast(RustTranscription, value) if callable(value) else None  # cast-ok: callable checked at binding


def _as_atranscription(value: object) -> RustAtranscription | None:
    return cast(RustAtranscription, value) if callable(value) else None  # cast-ok: callable checked at binding


TRANSCRIPTION: Final = COMPONENT.bind("transcription", validate=_as_transcription)
ATRANSCRIPTION: Final = COMPONENT.bind("atranscription", validate=_as_atranscription)


def configure_rust_transcription(
    *,
    transcription: RustTranscription | None | BindingUnset = BINDING_UNSET,
    atranscription: RustAtranscription | None | BindingUnset = BINDING_UNSET,
) -> None:
    TRANSCRIPTION.configure(transcription)
    ATRANSCRIPTION.configure(atranscription)


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
        native: Final = execution.select(TRANSCRIPTION)
        fallback: Final = (
            (lambda: function(*args, **kwargs))
            if execution.decision in (ExecutionDecision.PYTHON, ExecutionDecision.RUST_WITH_FALLBACK)
            else None
        )
        return invoke_lifecycle(
            execution=execution,
            native_call=(lambda: native(boundary_request, call_args, call_kwargs, HOST))
            if native is not None
            else None,
            python_fallback=fallback,
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
        native: Final = execution.select(ATRANSCRIPTION)
        fallback: Final = (
            (lambda: function(*args, **kwargs))
            if execution.decision in (ExecutionDecision.PYTHON, ExecutionDecision.RUST_WITH_FALLBACK)
            else None
        )
        return await ainvoke_lifecycle(
            execution=execution,
            native_call=(lambda: native(boundary_request, call_args, call_kwargs, HOST))
            if native is not None
            else None,
            python_fallback=fallback,
        )

    return wrapped
