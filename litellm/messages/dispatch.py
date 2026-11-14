import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator, Mapping
from types import MappingProxyType
from typing import Final, TypeAlias, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

from litellm.exceptions import BadRequestError, UnsupportedParamsError
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.anthropic.messages.native import handler as main
from litellm.llms.anthropic.messages.utils import litellm_logging_obj_from_kwargs
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.rust_bridge import catalog, detectors
from litellm.rust_bridge.catalog import Feature, Route, RouteContext, Target
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.dispatch import PublicDispatch
from litellm.rust_bridge.messages.entrypoints import (
    NATIVE_AMESSAGES,
    NATIVE_MESSAGES,
)
from litellm.rust_bridge.public_call import (
    AsyncFlag,
    NativeCall,
    bind,
    native_acall_hook,
    native_call,
    native_call_hook,
    optional_str,
    signature,
)
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from litellm.types.router import GenericLiteLLMParams

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
    """The native call takes one timeout, resolved from timeout, request_timeout and stream_timeout as Python's."""
    _, provider = _resolved_model(str(fields["model"]), optional_str(fields.get("custom_llm_provider")))
    timeout: Final = BaseLLMHTTPHandler._resolve_anthropic_messages_timeout(  # pyright: ignore[reportPrivateUsage]  # the handler's own timeout resolution
        litellm_params=GenericLiteLLMParams(
            timeout=kwargs.get("timeout"),
            request_timeout=kwargs.get("request_timeout"),
            stream_timeout=kwargs.get("stream_timeout"),
        ),
        stream=bool(fields.get("stream")),
        custom_llm_provider=provider or "",
    )
    if timeout is None:
        return kwargs
    return {**kwargs, "timeout": timeout}


def _public_request(
    legacy: inspect.Signature, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> NativeCall | None:
    fields: Final = bind(legacy, args, kwargs)
    if fields is None:
        return None
    model: Final = fields.get("model")
    if not isinstance(model, str):
        return None
    return native_call(legacy, args, _native_kwargs(fields, kwargs))


def _resolved_model(model: str, provider: str | None) -> tuple[str, str | None]:
    try:
        resolved_model, resolved_provider, _, _ = get_llm_provider(model, provider)
    except BadRequestError:
        return model, provider
    return resolved_model, resolved_provider


def _resolved(request: NativeCall) -> tuple[str, str | None]:
    return _resolved_model(str(request.resolved["model"]), optional_str(request.resolved.get("custom_llm_provider")))


def _route_target(
    requested_model: str,
    model: str,
    provider: str | None,
    model_info: object,
    api_base: str | None,
    compacting: bool,
) -> Target:
    native_config: Final = main._native_messages_config(model, provider, model_info, api_base)  # pyright: ignore[reportPrivateUsage]  # the handler's own passthrough lookup
    if native_config is not None:
        return Target.MESSAGES
    if not compacting and main._should_route_to_responses_api(provider, requested_model, model, model_info):  # pyright: ignore[reportPrivateUsage]  # the handler's own translation choice
        return Target.RESPONSES
    return Target.CHAT_COMPLETIONS


def _target(request: NativeCall, model: str, provider: str | None) -> Target:
    api_base: Final = request.resolved.get("api_base")
    return _route_target(
        str(request.resolved["model"]),
        model,
        provider,
        request.resolved.get("model_info"),
        api_base if isinstance(api_base, str) else None,
        compacting=request.resolved.get("compaction") is not None,
    )


def messages_target(model: str, custom_llm_provider: str | None, model_info: object, api_base: str | None) -> Target:
    resolved_model, resolved_provider = _resolved_model(model, custom_llm_provider)
    return _route_target(model, resolved_model, resolved_provider, model_info, api_base, compacting=False)


_NATIVE_COMMON_FEATURES: Final = frozenset(
    {
        Feature.TIMEOUT_VARIANTS,
        Feature.LATEST_ALIAS,
        Feature.MOCK_RESPONSE,
        Feature.CALLER_CLIENT,
        Feature.TLS_POLICY,
        Feature.AGENTIC_LOOP_HOOK,
        Feature.PROXY_MCP_TOOLS,
    }
)

_FEATURES: Final[detectors.Detectors] = MappingProxyType(
    {feature: detect for feature, detect in detectors.COMMON.items() if feature not in _NATIVE_COMMON_FEATURES}
)


def _context(request: NativeCall, features: detectors.Detectors) -> RouteContext:
    requested_model: Final = str(request.resolved["model"])
    model, provider = _resolved(request)
    return RouteContext(
        Route.MESSAGES,
        provider=provider,
        model=requested_model,
        target=_target(request, model, provider),
        features=detectors.detected(features, request),
    )


_DISPATCH: Final = PublicDispatch(
    route=Route.MESSAGES,
    request=lambda args, kwargs: _public_request(_MESSAGES, args, kwargs),
    context=lambda request: _context(request, _FEATURES),
)

_ADISPATCH: Final = PublicDispatch(
    route=Route.MESSAGES,
    request=lambda args, kwargs: _public_request(_AMESSAGES, args, kwargs),
    context=lambda request: _context(request, _FEATURES),
)


def _unsupported_features(features: detectors.Detectors, request: NativeCall) -> tuple[str, ...]:
    try:
        return tuple(sorted(feature.value for feature in detectors.detected(features, request)))
    except Exception:  # noqa: BLE001  # a failing detector names nothing, and the call is refused all the same
        return ()


def _checked_target(request: NativeCall, model: str, provider: str | None) -> Target | None:
    try:
        return _target(request, model, provider)
    except Exception:  # noqa: BLE001  # the provider alone still decides whether the call has a Python path
        return None


def _rust_only_refusal(
    legacy: inspect.Signature, features: detectors.Detectors, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> UnsupportedParamsError | None:
    if bind(legacy, args, kwargs) is None:
        return None
    request: Final = native_call(legacy, args, kwargs)
    requested_model: Final = request.resolved.get("model")
    if not isinstance(requested_model, str):
        return None
    model, provider = _resolved(request)
    context: Final = RouteContext(
        Route.MESSAGES, provider=provider, model=requested_model, target=_checked_target(request, model, provider)
    )
    if catalog.rollout(context) is not Rollout.RUST_REQUIRED:
        return None
    unsupported: Final = _unsupported_features(features, request)
    return UnsupportedParamsError(
        message=(
            f"provider {provider} on /v1/messages runs only on the Rust path, which does not support "
            f"{', '.join(unsupported) if unsupported else 'this call'}"
        ),
        llm_provider=provider,
        model=requested_model,
    )


def _python_messages_handler(
    *args: object,
    **kwargs: object,  # kwargs-ok: forwards the public call shape
) -> MessagesResult | Coroutine[object, object, MessagesResult]:
    refusal: Final = _rust_only_refusal(_MESSAGES, _FEATURES, args, kwargs)
    if refusal is not None:
        raise refusal
    return _PYTHON_MESSAGES(*args, **kwargs)


async def _python_anthropic_messages(*args: object, **kwargs: object) -> MessagesResult:  # kwargs-ok: public call shape
    refusal: Final = _rust_only_refusal(_AMESSAGES, _FEATURES, args, kwargs)
    if refusal is not None:
        raise refusal
    return await _PYTHON_AMESSAGES(*args, **kwargs)


def _forget_earlier_attempt_bridge(kwargs: Mapping[str, object]) -> None:
    logging_obj: Final = litellm_logging_obj_from_kwargs(kwargs)
    if logging_obj is not None:
        logging_obj.clear_bridged_model_response()


def anthropic_messages_handler(
    *args: object,
    **kwargs: object,  # kwargs-ok: preserve the public Anthropic Messages call shape
) -> MessagesResult | Coroutine[object, object, MessagesResult]:
    if _FLAG.requested(kwargs):
        return anthropic_messages(**_FLAG.keywords(args, kwargs))
    _forget_earlier_attempt_bridge(kwargs)
    return _DISPATCH.run(
        args,
        _FLAG.dropped(kwargs),
        python=_python_messages_handler,
        binding=NATIVE_MESSAGES,
        native=native_call_hook,
    )


async def anthropic_messages(*args: object, **kwargs: object) -> MessagesResult:  # kwargs-ok: public call shape
    _forget_earlier_attempt_bridge(kwargs)
    return await _ADISPATCH.arun(
        args,
        _FLAG.dropped(kwargs),
        python=_python_anthropic_messages,
        binding=NATIVE_AMESSAGES,
        native=native_acall_hook,
    )


anthropic_messages_handler.__doc__ = _PYTHON_MESSAGES.__doc__
anthropic_messages_handler.__wrapped__ = _PYTHON_MESSAGES  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
anthropic_messages.__doc__ = _PYTHON_AMESSAGES.__doc__
anthropic_messages.__wrapped__ = _PYTHON_AMESSAGES  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
