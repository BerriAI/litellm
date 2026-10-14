import inspect
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import Final, TypeAlias, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

from litellm.responses import main
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.rust_bridge.catalog import Route, RouteContext
from litellm.rust_bridge.dispatch import AsyncPublicRoute, PublicDispatch, PublicRoute
from litellm.rust_bridge.public_call import (
    NativeCall,
    bind,
    native_call,
    native_call_hook,
    optional_str,
    signature,
)
from litellm.rust_bridge.responses.entrypoints import (
    NATIVE_ARESPONSES,
    NATIVE_RESPONSES,
)
from litellm.types.llms.openai import ResponsesAPIResponse

__all__ = ("aresponses", "responses")

ResponsesResult: TypeAlias = ResponsesAPIResponse | BaseResponsesAPIStreamingIterator
PythonResponses: TypeAlias = Callable[..., ResponsesResult | Coroutine[object, object, ResponsesResult]]
PythonAresponses: TypeAlias = Callable[..., Awaitable[ResponsesResult]]


def _python_responses() -> PythonResponses:
    return cast(  # cast-ok: forward the original call shape through the Python @client decorator
        PythonResponses,
        main.responses,  # noqa: TID251  # dispatch boundary owns this Python fallback
    )


def _python_aresponses() -> PythonAresponses:
    return cast(  # cast-ok: forward the original call shape through the Python @client decorator
        PythonAresponses,
        main.aresponses,  # noqa: TID251  # dispatch boundary owns this Python fallback
    )


_PYTHON_RESPONSES: Final = _python_responses()
_RESPONSES: Final = signature(_PYTHON_RESPONSES)
_PYTHON_ARESPONSES: Final = _python_aresponses()
_ARESPONSES: Final = signature(_PYTHON_ARESPONSES)


def _public_request(
    legacy: inspect.Signature, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> NativeCall | None:
    fields: Final = bind(legacy, args, kwargs)
    if fields is None:
        return None
    model: Final = fields.get("model")
    if not isinstance(model, str):
        return None
    return native_call(legacy, args, kwargs)


def _context(request: NativeCall) -> RouteContext:
    return RouteContext(
        Route.RESPONSES,
        provider=optional_str(request.resolved.get("custom_llm_provider")),
        model=str(request.resolved["model"]),
    )


_DISPATCH: Final = PublicDispatch(
    route=Route.RESPONSES,
    request=lambda args, kwargs: _public_request(_RESPONSES, args, kwargs),
    context=_context,
    bypass=lambda request: request.kwargs.get("aresponses") is True,
)

_ADISPATCH: Final = PublicDispatch(
    route=Route.RESPONSES,
    request=lambda args, kwargs: _public_request(_ARESPONSES, args, kwargs),
    context=_context,
)

RESPONSES: Final = PublicRoute(
    dispatch=_DISPATCH,
    binding=NATIVE_RESPONSES,
    native=native_call_hook,
    python=_PYTHON_RESPONSES,
)

ARESPONSES: Final = AsyncPublicRoute(
    dispatch=_ADISPATCH,
    binding=NATIVE_ARESPONSES,
    native=native_call_hook,
    python=_PYTHON_ARESPONSES,
)


def responses(
    *args: object,
    **kwargs: object,  # kwargs-ok: preserve the public Responses call shape
) -> ResponsesResult | Coroutine[object, object, ResponsesResult]:
    return RESPONSES.run(args, kwargs)


async def aresponses(*args: object, **kwargs: object) -> ResponsesResult:  # kwargs-ok: preserve the public call shape
    return await ARESPONSES.arun(args, kwargs)


responses.__doc__ = _PYTHON_RESPONSES.__doc__
responses.__wrapped__ = _PYTHON_RESPONSES  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
aresponses.__doc__ = _PYTHON_ARESPONSES.__doc__
aresponses.__wrapped__ = _PYTHON_ARESPONSES  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
