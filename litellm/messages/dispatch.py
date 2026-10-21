import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator, Mapping
from typing import Final, TypeAlias, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.anthropic.pass_through.messages import handler as main
from litellm.rust_bridge.catalog import Route, RouteContext
from litellm.rust_bridge.dispatch import PublicDispatch
from litellm.rust_bridge.messages.entrypoints import (
    NATIVE_AMESSAGES,
    NATIVE_MESSAGES,
)
from litellm.rust_bridge.public_call import (
    NativeCall,
    bind,
    native_call,
    native_call_hook,
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
) -> NativeCall | None:
    fields: Final = bind(legacy, args, kwargs)
    if fields is None:
        return None
    model: Final = fields.get("model")
    messages: Final = optional_sequence(fields.get("messages"))
    max_tokens: Final = fields.get("max_tokens")
    if not isinstance(model, str) or messages is None or not isinstance(max_tokens, int):
        return None
    return native_call(legacy, args, kwargs)


def _resolved_provider(request: NativeCall) -> str | None:
    try:
        return get_llm_provider(
            str(request.resolved["model"]), optional_str(request.resolved.get("custom_llm_provider"))
        )[1]
    except BadRequestError:
        return optional_str(request.resolved.get("custom_llm_provider"))


def _context(request: NativeCall) -> RouteContext:
    return RouteContext(
        Route.MESSAGES,
        provider=_resolved_provider(request),
        model=str(request.resolved["model"]),
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
        native=native_call_hook,
    )


async def anthropic_messages(*args: object, **kwargs: object) -> MessagesResult:  # kwargs-ok: public call shape
    python: Final = _PYTHON_AMESSAGES
    return await _ADISPATCH.arun(
        args,
        kwargs,
        python=python,
        binding=NATIVE_AMESSAGES,
        native=native_call_hook,
    )


anthropic_messages_handler.__doc__ = _PYTHON_MESSAGES.__doc__
anthropic_messages_handler.__wrapped__ = _PYTHON_MESSAGES  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
anthropic_messages.__doc__ = _PYTHON_AMESSAGES.__doc__
anthropic_messages.__wrapped__ = _PYTHON_AMESSAGES  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
