import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator, Mapping
from types import MappingProxyType
from typing import Final, TypeAlias, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

from litellm.llms.anthropic.experimental_pass_through.messages import handler as main
from litellm.rust_bridge.catalog import Context, Delivery, Route
from litellm.rust_bridge.dispatch import PublicDispatch, call_hook
from litellm.rust_bridge.messages.entrypoints import (
    NATIVE_AMESSAGES,
    NATIVE_MESSAGES,
    LiteLLMMessagesRequest,
)
from litellm.rust_bridge.public_call import (
    bind,
    optional_bool,
    optional_mapping,
    optional_sequence,
    optional_str,
    signature,
)
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

__all__ = ("anthropic_messages", "anthropic_messages_handler")

MessagesResult: TypeAlias = AnthropicMessagesResponse | Iterator[bytes] | AsyncIterator[object]
PythonMessages: TypeAlias = Callable[..., MessagesResult | Coroutine[object, object, MessagesResult]]
PythonAmessages: TypeAlias = Callable[..., Awaitable[MessagesResult]]


def _python_messages() -> PythonMessages:
    return cast(  # cast-ok: forward the original call shape through the legacy handler
        PythonMessages,
        main.anthropic_messages_handler,  # noqa: TID251  # dispatch boundary owns this Python fallback
    )


def _python_amessages() -> PythonAmessages:
    return cast(  # cast-ok: forward the original call shape through the Python @client decorator
        PythonAmessages,
        main.anthropic_messages,  # noqa: TID251  # dispatch boundary owns this Python fallback
    )


_PYTHON_MESSAGES: Final = _python_messages()
_MESSAGES: Final = signature(_PYTHON_MESSAGES)
_PYTHON_AMESSAGES: Final = _python_amessages()
_AMESSAGES: Final = signature(_PYTHON_AMESSAGES)


def _public_request(
    legacy: inspect.Signature, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> LiteLLMMessagesRequest | None:
    fields: Final = bind(legacy, args, kwargs)
    if fields is None:
        return None
    model: Final = fields.get("model")
    messages: Final = optional_sequence(fields.get("messages"))
    max_tokens: Final = fields.get("max_tokens")
    if not isinstance(model, str) or messages is None or not isinstance(max_tokens, int):
        return None
    return LiteLLMMessagesRequest(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=optional_bool(fields.get("stream")),
        api_key=optional_str(fields.get("api_key")),
        api_base=optional_str(fields.get("api_base")),
        custom_llm_provider=optional_str(fields.get("custom_llm_provider")),
        kwargs=optional_mapping(fields.get("kwargs")) or MappingProxyType({}),
    )


def _context(request: LiteLLMMessagesRequest) -> Context:
    return Context(
        Route.MESSAGES,
        provider=request.custom_llm_provider,
        model=request.model,
        delivery=Delivery.STREAMING if request.stream else Delivery.COMPLETED,
    )


_DISPATCH: Final = PublicDispatch(
    route=Route.MESSAGES,
    request=lambda args, kwargs: _public_request(_MESSAGES, args, kwargs),
    context=_context,
    bypass=lambda request: request.kwargs.get("is_async") is True,
)

_ADISPATCH: Final = PublicDispatch(
    route=Route.MESSAGES,
    request=lambda args, kwargs: _public_request(_AMESSAGES, args, kwargs),
    context=_context,
)


def anthropic_messages_handler(
    *args: object,
    **kwargs: object,  # kwargs-ok: preserve the public Anthropic Messages call shape
) -> MessagesResult | Coroutine[object, object, MessagesResult]:
    python: Final = _PYTHON_MESSAGES
    return _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=NATIVE_MESSAGES,
        native=call_hook,
    )


async def anthropic_messages(*args: object, **kwargs: object) -> MessagesResult:  # kwargs-ok: public call shape
    python: Final = _PYTHON_AMESSAGES
    return await _ADISPATCH.arun(
        args,
        kwargs,
        python=python,
        binding=NATIVE_AMESSAGES,
        native=call_hook,
    )


anthropic_messages_handler.__doc__ = _PYTHON_MESSAGES.__doc__
anthropic_messages_handler.__wrapped__ = _PYTHON_MESSAGES  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
anthropic_messages.__doc__ = _PYTHON_AMESSAGES.__doc__
anthropic_messages.__wrapped__ = _PYTHON_AMESSAGES  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
