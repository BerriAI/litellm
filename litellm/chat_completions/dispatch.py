import inspect
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import Final, TypeAlias, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

from litellm import main
from litellm.rust_bridge.catalog import Route, RouteContext
from litellm.rust_bridge.chat_completions.entrypoints import (
    NATIVE_ACOMPLETION,
    NATIVE_COMPLETION,
)
from litellm.rust_bridge.dispatch import PublicDispatch
from litellm.rust_bridge.public_call import (
    NativeCall,
    bind,
    native_call,
    native_call_hook,
    optional_sequence,
    optional_str,
    signature,
)
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

__all__ = ("acompletion", "completion")

ChatResult: TypeAlias = ModelResponse | CustomStreamWrapper
PythonCompletion: TypeAlias = Callable[..., ChatResult | Coroutine[object, object, ChatResult]]
PythonAcompletion: TypeAlias = Callable[..., Awaitable[ChatResult]]


def _python_completion() -> PythonCompletion:
    return cast(  # cast-ok: forward the original call shape through the Python @client decorator
        PythonCompletion,
        main.completion,  # noqa: TID251  # dispatch boundary owns this Python fallback
    )


def _python_acompletion() -> PythonAcompletion:
    return cast(  # cast-ok: forward the original call shape through the Python @client decorator
        PythonAcompletion,
        main.acompletion,  # noqa: TID251  # dispatch boundary owns this Python fallback
    )


_PYTHON_COMPLETION: Final = _python_completion()
_COMPLETION: Final = signature(_PYTHON_COMPLETION)
_PYTHON_ACOMPLETION: Final = _python_acompletion()
_ACOMPLETION: Final = signature(_PYTHON_ACOMPLETION)


def _public_request(
    legacy: inspect.Signature, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> NativeCall | None:
    fields: Final = bind(legacy, args, kwargs)
    if fields is None:
        return None
    model: Final = fields.get("model")
    messages: Final = optional_sequence(fields.get("messages"))
    if not isinstance(model, str) or messages is None:
        return None
    return native_call(args, kwargs, fields)


def _context(request: NativeCall) -> RouteContext:
    return RouteContext(
        Route.CHAT_COMPLETIONS,
        provider=optional_str(request.bound.get("custom_llm_provider")),
        model=str(request.bound["model"]),
    )


_DISPATCH: Final = PublicDispatch(
    route=Route.CHAT_COMPLETIONS,
    request=lambda args, kwargs: _public_request(_COMPLETION, args, kwargs),
    context=_context,
    bypass=lambda request: request.kwargs.get("acompletion") is True,
)

_ADISPATCH: Final = PublicDispatch(
    route=Route.CHAT_COMPLETIONS,
    request=lambda args, kwargs: _public_request(_ACOMPLETION, args, kwargs),
    context=_context,
)


def completion(
    *args: object,
    **kwargs: object,  # kwargs-ok: preserve the public chat completions call shape
) -> ChatResult | Coroutine[object, object, ChatResult]:
    python: Final = _PYTHON_COMPLETION
    return _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=NATIVE_COMPLETION,
        native=native_call_hook,
    )


async def acompletion(*args: object, **kwargs: object) -> ChatResult:  # kwargs-ok: preserve the public call shape
    python: Final = _PYTHON_ACOMPLETION
    return await _ADISPATCH.arun(
        args,
        kwargs,
        python=python,
        binding=NATIVE_ACOMPLETION,
        native=native_call_hook,
    )


completion.__doc__ = _PYTHON_COMPLETION.__doc__
completion.__wrapped__ = _PYTHON_COMPLETION  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
acompletion.__doc__ = _PYTHON_ACOMPLETION.__doc__
acompletion.__wrapped__ = _PYTHON_ACOMPLETION  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
