import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator, Mapping
from types import MappingProxyType
from typing import Final, TypeAlias, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.anthropic.pass_through.messages import handler as main
from litellm.llms.anthropic.pass_through.messages.interceptors import get_messages_interceptors
from litellm.llms.anthropic.pass_through.safeguards import native_route_classifies
from litellm.rust_bridge import detectors
from litellm.rust_bridge.catalog import Feature, Route, RouteContext, Target
from litellm.rust_bridge.dispatch import PublicDispatch
from litellm.rust_bridge.messages.entrypoints import (
    NATIVE_AMESSAGES,
    NATIVE_MESSAGES,
)
from litellm.rust_bridge.public_call import (
    AsyncFlag,
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
_FLAG: Final = AsyncFlag("is_async", sync=_MESSAGES)
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


def _resolved(request: NativeCall) -> tuple[str, str | None]:
    model: Final = str(request.resolved["model"])
    provider: Final = optional_str(request.resolved.get("custom_llm_provider"))
    try:
        resolved_model, resolved_provider, _, _ = get_llm_provider(model, provider)
    except BadRequestError:
        return model, provider
    return resolved_model, resolved_provider


def _target(request: NativeCall, model: str, provider: str | None) -> Target:
    requested_model: Final = str(request.resolved["model"])
    if main._native_messages_config(model, provider, request.resolved.get("model_info")) is not None:  # pyright: ignore[reportPrivateUsage]  # the handler's own passthrough lookup
        return Target.MESSAGES
    compacting: Final = request.resolved.get("compaction") is not None
    if not compacting and main._should_route_to_responses_api(provider, requested_model, model):  # pyright: ignore[reportPrivateUsage]  # the handler's own translation choice
        return Target.RESPONSES
    return Target.CHAT_COMPLETIONS


def _intercepted(request: NativeCall) -> bool:
    tools: Final = optional_sequence(request.resolved.get("tools"))
    if not tools:
        return False
    tool_dicts: Final[list[dict[str, object]]] = [tool for tool in tools if isinstance(tool, dict)]
    provider: Final = _resolved(request)[1]
    return any(interceptor.can_handle(tool_dicts, provider) for interceptor in get_messages_interceptors())


_FEATURES: Final[detectors.Detectors] = MappingProxyType(
    {
        **detectors.COMMON,
        Feature.SAFEGUARDS_CLASSIFIER: lambda request: native_route_classifies(
            str(request.resolved["model"]), request.resolved.get("safeguards")
        ),
        Feature.PRE_REQUEST_HOOK: detectors.pre_request_hook,
        Feature.WEBSEARCH_INTERCEPTION: detectors.websearch_interception,
        Feature.MESSAGES_INTERCEPTOR: _intercepted,
        Feature.RESPONSE_CACHE: detectors.response_cache,
    }
)


def _context(request: NativeCall) -> RouteContext:
    requested_model: Final = str(request.resolved["model"])
    model, provider = _resolved(request)
    return RouteContext(
        Route.MESSAGES,
        provider=provider,
        model=requested_model,
        target=_target(request, model, provider),
        features=detectors.detected(_FEATURES, request),
    )


_DISPATCH: Final = PublicDispatch(
    route=Route.MESSAGES,
    request=lambda args, kwargs: _public_request(_MESSAGES, args, kwargs),
    context=_context,
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
    if _FLAG.requested(kwargs):
        return anthropic_messages(**_FLAG.keywords(args, kwargs))
    return _DISPATCH.run(
        args,
        _FLAG.dropped(kwargs),
        python=python,
        binding=NATIVE_MESSAGES,
        native=native_call_hook,
    )


async def anthropic_messages(*args: object, **kwargs: object) -> MessagesResult:  # kwargs-ok: public call shape
    python: Final = _PYTHON_AMESSAGES
    return await _ADISPATCH.arun(
        args,
        _FLAG.dropped(kwargs),
        python=python,
        binding=NATIVE_AMESSAGES,
        native=native_call_hook,
    )


anthropic_messages_handler.__doc__ = _PYTHON_MESSAGES.__doc__
anthropic_messages_handler.__wrapped__ = _PYTHON_MESSAGES  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
anthropic_messages.__doc__ = _PYTHON_AMESSAGES.__doc__
anthropic_messages.__wrapped__ = _PYTHON_AMESSAGES  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
