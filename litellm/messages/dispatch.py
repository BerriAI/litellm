import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator, Mapping
from types import MappingProxyType
from typing import Final, TypeAlias, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

from litellm.llms.anthropic.pass_through.messages import handler as main
from litellm.llms.anthropic.pass_through.messages.interceptors import get_messages_interceptors
from litellm.llms.anthropic.pass_through.safeguards import native_route_classifies
from litellm.rust_bridge import detectors, provider
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
    optional_mapping,
    optional_sequence,
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
_FLAG: Final = AsyncFlag("is_async", sync=_MESSAGES, is_set=bool)
_PYTHON_AMESSAGES: Final = _python_amessages()
_AMESSAGES: Final = signature(_PYTHON_AMESSAGES)


def _native_kwargs(fields: Mapping[str, object], kwargs: Mapping[str, object]) -> Mapping[str, object]:
    stream_timeout: Final = kwargs.get("stream_timeout")
    if fields.get("stream") is not True or stream_timeout is None:
        return kwargs
    return {**kwargs, "timeout": stream_timeout}


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
    return native_call(legacy, args, _native_kwargs(fields, kwargs))


def _target(request: NativeCall, model: str, provider: str | None) -> Target:
    requested_model: Final = str(request.resolved["model"])
    model_info: Final = request.resolved.get("model_info")
    api_base: Final = request.resolved.get("api_base")
    native_config: Final = main._native_messages_config(  # pyright: ignore[reportPrivateUsage]  # the handler's own passthrough lookup
        model, provider, model_info, api_base if isinstance(api_base, str) else None
    )
    if native_config is not None:
        return Target.MESSAGES
    compacting: Final = request.resolved.get("compaction") is not None
    if not compacting and main._should_route_to_responses_api(provider, requested_model, model, model_info):  # pyright: ignore[reportPrivateUsage]  # the handler's own translation choice
        return Target.RESPONSES
    return Target.CHAT_COMPLETIONS


def _intercepted(request: NativeCall, provider: str | None) -> bool:
    tools: Final = optional_sequence(request.resolved.get("tools"))
    if not tools:
        return False
    mappings: Final = tuple(mapping for mapping in map(optional_mapping, tools) if mapping is not None)
    return any(interceptor.can_handle(mappings, provider) for interceptor in get_messages_interceptors())


_FEATURES: Final[detectors.Detectors] = MappingProxyType(
    {
        **detectors.COMMON,
        Feature.SAFEGUARDS_CLASSIFIER: lambda request: native_route_classifies(
            str(request.resolved["model"]), request.resolved.get("safeguards")
        ),
        Feature.RESPONSE_CACHE: detectors.response_cache,
        Feature.MODIFY_PARAMS: detectors.modify_params,
        Feature.WORKLOAD_IDENTITY: detectors.anthropic_workload_identity,
        Feature.CACHE_CONTROL_INJECTION: detectors.cache_control_injection,
        Feature.STREAM_COST_INJECTION: detectors.stream_cost_injection,
        Feature.THINKING_SIGNATURE_RETRY: detectors.thinking_signature_retry,
    }
)

_ASYNC_FEATURES: Final[detectors.Detectors] = MappingProxyType(
    {
        **_FEATURES,
        Feature.PRE_REQUEST_HOOK: detectors.pre_request_hook,
        Feature.WEBSEARCH_INTERCEPTION: detectors.websearch_interception,
    }
)


def _features(
    request: NativeCall, provider: str | None, features: detectors.Detectors, intercepting: bool
) -> frozenset[Feature]:
    detected: Final = detectors.detected(features, request)
    if not intercepting or not _intercepted(request, provider):
        return detected
    return detected | {Feature.MESSAGES_INTERCEPTOR}


def _context(request: NativeCall, features: detectors.Detectors, intercepting: bool) -> RouteContext:
    requested_model: Final = str(request.resolved["model"])
    resolution: Final = provider.resolve(request)
    match resolution:
        case provider.Resolved(model=resolved_model, provider=resolved_provider):
            return RouteContext(
                Route.MESSAGES,
                provider=resolved_provider,
                model=requested_model,
                target=_target(request, resolved_model, resolved_provider),
                features=_features(request, resolved_provider, features, intercepting),
            )
        case provider.Unresolved():
            return RouteContext(
                Route.MESSAGES, model=requested_model, features=_features(request, None, features, intercepting)
            )


_DISPATCH: Final = PublicDispatch(
    route=Route.MESSAGES,
    request=lambda args, kwargs: _public_request(_MESSAGES, args, kwargs),
    context=lambda request: _context(request, _FEATURES, intercepting=False),
)

_ADISPATCH: Final = PublicDispatch(
    route=Route.MESSAGES,
    request=lambda args, kwargs: _public_request(_AMESSAGES, args, kwargs),
    context=lambda request: _context(request, _ASYNC_FEATURES, intercepting=True),
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
