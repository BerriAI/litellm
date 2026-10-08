from collections.abc import Awaitable, Callable, Coroutine
from typing import Final, TypeAlias, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

from litellm import main
from litellm.rust_bridge.catalog import Route
from litellm.rust_bridge.chat_completions.entrypoints import (
    NATIVE_ACOMPLETION,
    NATIVE_COMPLETION,
)
from litellm.rust_bridge.dispatch import Fields, PublicDispatch, model_is_named
from litellm.rust_bridge.public_call import binder, native_call_hook, optional_sequence
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
_PYTHON_ACOMPLETION: Final = _python_acompletion()


def _chat_fields(fields: Fields) -> bool:
    return model_is_named(fields) and optional_sequence(fields.get("messages")) is not None


_DISPATCH: Final = PublicDispatch(
    Route.CHAT_COMPLETIONS, bind=binder(_PYTHON_COMPLETION), internal_hop="acompletion", accepts=_chat_fields
)
_ADISPATCH: Final = PublicDispatch(Route.CHAT_COMPLETIONS, bind=binder(_PYTHON_ACOMPLETION), accepts=_chat_fields)


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
