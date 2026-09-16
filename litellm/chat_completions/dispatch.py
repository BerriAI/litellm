import inspect
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from types import MappingProxyType
from typing import Final, TypeAlias, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

from litellm import main
from litellm.rust_bridge.catalog import Context, Delivery, Route
from litellm.rust_bridge.chat_completions.entrypoints import (
    NATIVE_ACOMPLETION,
    NATIVE_COMPLETION,
    LiteLLMChatCompletionsRequest,
    NativeAcompletion,
)
from litellm.rust_bridge.public_call import (
    bind,
    optional_bool,
    optional_mapping,
    optional_sequence,
    optional_str,
    signature,
)
from litellm.rust_bridge.runtime import arun, run
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

__all__ = ("acompletion", "completion")

ChatResult: TypeAlias = ModelResponse | CustomStreamWrapper
PythonCompletion: TypeAlias = Callable[..., ChatResult | Coroutine[object, object, ChatResult]]
PythonAcompletion: TypeAlias = Callable[..., Awaitable[ChatResult]]


def _python_completion() -> PythonCompletion:
    return cast(  # cast-ok: forward the original call shape through the Python @client decorator
        PythonCompletion, main.completion
    )


def _python_acompletion() -> PythonAcompletion:
    return cast(  # cast-ok: forward the original call shape through the Python @client decorator
        PythonAcompletion, main.acompletion
    )


_COMPLETION: Final = signature(_python_completion())
_ACOMPLETION: Final = signature(_python_acompletion())


def _public_request(
    legacy: inspect.Signature, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> LiteLLMChatCompletionsRequest | None:
    fields: Final = bind(legacy, args, kwargs)
    if fields is None:
        return None
    model: Final = fields.get("model")
    messages: Final = optional_sequence(fields.get("messages"))
    extra: Final = optional_mapping(fields.get("kwargs")) or MappingProxyType({})
    if not isinstance(model, str) or messages is None:
        return None
    return LiteLLMChatCompletionsRequest(
        model=model,
        messages=messages,
        stream=optional_bool(fields.get("stream")),
        api_key=optional_str(fields.get("api_key")),
        api_base=optional_str(extra.get("api_base")) or optional_str(fields.get("base_url")),
        custom_llm_provider=optional_str(extra.get("custom_llm_provider")),
        extra_headers=optional_mapping(fields.get("extra_headers")),
        kwargs=extra,
    )


def completion(
    *args: object,
    **kwargs: object,  # kwargs-ok: preserve the public chat completions call shape
) -> ChatResult | Coroutine[object, object, ChatResult]:
    python: Final = _python_completion()
    request: Final = _public_request(_COMPLETION, args, kwargs)
    if request is None or request.kwargs.get("acompletion") is True:
        return python(*args, **kwargs)
    return run(
        _context(request),
        binding=NATIVE_COMPLETION,
        native=lambda hook: hook(request, args, kwargs),
        python=lambda: python(*args, **kwargs),
    )


async def acompletion(*args: object, **kwargs: object) -> ChatResult:  # kwargs-ok: preserve the public call shape
    python: Final = _python_acompletion()
    request: Final = _public_request(_ACOMPLETION, args, kwargs)
    if request is None:
        return await python(*args, **kwargs)

    async def native(hook: NativeAcompletion) -> ChatResult:
        return await hook(request, args, kwargs)

    return await arun(
        _context(request), binding=NATIVE_ACOMPLETION, native=native, python=lambda: python(*args, **kwargs)
    )


def _context(request: LiteLLMChatCompletionsRequest) -> Context:
    return Context(
        Route.CHAT_COMPLETIONS,
        provider=request.custom_llm_provider,
        model=request.model,
        delivery=Delivery.STREAMING if request.stream else Delivery.COMPLETED,
    )


completion.__doc__ = _python_completion().__doc__
completion.__wrapped__ = _python_completion()  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
acompletion.__doc__ = _python_acompletion().__doc__
acompletion.__wrapped__ = _python_acompletion()  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
