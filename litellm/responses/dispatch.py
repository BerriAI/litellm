import inspect
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from types import MappingProxyType
from typing import Final, TypeAlias, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

from litellm.responses import main
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.rust_bridge import detectors, provider
from litellm.rust_bridge.catalog import Feature, Route, RouteContext
from litellm.rust_bridge.dispatch import PublicDispatch
from litellm.rust_bridge.public_call import (
    AsyncFlag,
    NativeCall,
    bind,
    native_call,
    native_call_hook,
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
_FLAG: Final = AsyncFlag("aresponses", sync=_RESPONSES)
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


_FEATURES: Final[detectors.Detectors] = MappingProxyType(
    {
        **detectors.COMMON,
        Feature.DROP_PARAMS: detectors.drop_params,
        Feature.MODIFY_PARAMS: detectors.modify_params,
        Feature.STREAMING: detectors.streaming,
        Feature.PROMPT_MANAGEMENT: detectors.prompt_management,
        Feature.TEXT_FORMAT: detectors.text_format,
        Feature.CHAT_COMPLETIONS_BRIDGE: detectors.chat_completions_bridge,
        Feature.RETRIES_FALLBACKS: detectors.retries_fallbacks,
        Feature.DEPLOYMENT_RESPONSE_ID: detectors.deployment_response_id,
        Feature.ENCODED_ITEM_IDS: detectors.encoded_item_ids,
        Feature.INPUT_REWRITE: detectors.input_rewrite,
        Feature.EXTRA_BODY: detectors.extra_body,
        Feature.PROVIDER_AFFINITY: detectors.provider_affinity,
        Feature.RESPONSE_CACHE: detectors.response_cache,
        Feature.MANAGED_RESOURCE_IDS: detectors.managed_resource_ids,
        Feature.CACHE_CONTROL_MARKS: detectors.cache_control_marks,
        Feature.ANTHROPIC_TOOL_SEARCH: detectors.anthropic_tool_search,
        Feature.TOOL_SCHEMA_REPAIR: detectors.tool_schema_repair,
        Feature.MODEL_PARAM_GATING: detectors.model_param_gating,
        Feature.WORKLOAD_IDENTITY: detectors.openai_workload_identity,
    }
)


def _context(request: NativeCall) -> RouteContext:
    resolution: Final = provider.resolve(request)
    return RouteContext(
        Route.RESPONSES,
        provider=resolution.provider if isinstance(resolution, provider.Resolved) else None,
        model=str(request.resolved["model"]),
        features=detectors.detected(_FEATURES, request),
    )


_DISPATCH: Final = PublicDispatch(
    route=Route.RESPONSES,
    request=lambda args, kwargs: _public_request(_RESPONSES, args, kwargs),
    context=_context,
)

_ADISPATCH: Final = PublicDispatch(
    route=Route.RESPONSES,
    request=lambda args, kwargs: _public_request(_ARESPONSES, args, kwargs),
    context=_context,
)


def responses(
    *args: object,
    **kwargs: object,  # kwargs-ok: preserve the public Responses call shape
) -> ResponsesResult | Coroutine[object, object, ResponsesResult]:
    python: Final = _PYTHON_RESPONSES
    if _FLAG.requested(kwargs):
        return aresponses(**_FLAG.keywords(args, kwargs))
    return _DISPATCH.run(
        args,
        _FLAG.dropped(kwargs),
        python=python,
        binding=NATIVE_RESPONSES,
        native=native_call_hook,
    )


async def aresponses(*args: object, **kwargs: object) -> ResponsesResult:  # kwargs-ok: preserve the public call shape
    python: Final = _PYTHON_ARESPONSES
    return await _ADISPATCH.arun(
        args,
        _FLAG.dropped(kwargs),
        python=python,
        binding=NATIVE_ARESPONSES,
        native=native_call_hook,
    )


responses.__doc__ = _PYTHON_RESPONSES.__doc__
responses.__wrapped__ = _PYTHON_RESPONSES  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
aresponses.__doc__ = _PYTHON_ARESPONSES.__doc__
aresponses.__wrapped__ = _PYTHON_ARESPONSES  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
