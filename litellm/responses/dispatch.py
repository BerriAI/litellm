import inspect
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from types import MappingProxyType
from typing import Final, TypeAlias, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

from litellm.responses import main
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.rust_bridge.catalog import Context, Delivery, Route
from litellm.rust_bridge.dispatch import PublicDispatch
from litellm.rust_bridge.public_call import bind, optional_bool, optional_mapping, optional_str, signature
from litellm.rust_bridge.responses.entrypoints import (
    NATIVE_ARESPONSES,
    NATIVE_RESPONSES,
    LiteLLMResponsesRequest,
)
from litellm.types.llms.openai import ResponsesAPIResponse

__all__ = ("aresponses", "responses")

ResponsesResult: TypeAlias = ResponsesAPIResponse | BaseResponsesAPIStreamingIterator
PythonResponses: TypeAlias = Callable[..., ResponsesResult | Coroutine[object, object, ResponsesResult]]
PythonAresponses: TypeAlias = Callable[..., Awaitable[ResponsesResult]]


def _python_responses() -> PythonResponses:
    return cast(  # cast-ok: forward the original call shape through the Python @client decorator
        PythonResponses, main.responses
    )


def _python_aresponses() -> PythonAresponses:
    return cast(  # cast-ok: forward the original call shape through the Python @client decorator
        PythonAresponses, main.aresponses
    )


_RESPONSES: Final = signature(_python_responses())
_ARESPONSES: Final = signature(_python_aresponses())


def _public_request(
    legacy: inspect.Signature, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> LiteLLMResponsesRequest | None:
    fields: Final = bind(legacy, args, kwargs)
    if fields is None:
        return None
    model: Final = fields.get("model")
    extra: Final = optional_mapping(fields.get("kwargs")) or MappingProxyType({})
    if not isinstance(model, str):
        return None
    return LiteLLMResponsesRequest(
        model=model,
        input=fields.get("input"),
        stream=optional_bool(fields.get("stream")),
        api_key=optional_str(extra.get("api_key")),
        api_base=optional_str(extra.get("api_base")) or optional_str(extra.get("base_url")),
        custom_llm_provider=optional_str(fields.get("custom_llm_provider")),
        extra_headers=optional_mapping(fields.get("extra_headers")),
        kwargs=extra,
    )


_DISPATCH: Final = PublicDispatch(
    route=Route.RESPONSES,
    request=lambda args, kwargs: _public_request(_RESPONSES, args, kwargs),
    context=lambda request: _context(request),
    bypass=lambda request: request.kwargs.get("aresponses") is True,
)

_ADISPATCH: Final = PublicDispatch(
    route=Route.RESPONSES,
    request=lambda args, kwargs: _public_request(_ARESPONSES, args, kwargs),
    context=lambda request: _context(request),
)


def responses(
    *args: object,
    **kwargs: object,  # kwargs-ok: preserve the public Responses call shape
) -> ResponsesResult | Coroutine[object, object, ResponsesResult]:
    python: Final = _python_responses()
    return _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=NATIVE_RESPONSES,
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
    )


async def aresponses(*args: object, **kwargs: object) -> ResponsesResult:  # kwargs-ok: preserve the public call shape
    python: Final = _python_aresponses()
    return await _ADISPATCH.arun(
        args,
        kwargs,
        python=python,
        binding=NATIVE_ARESPONSES,
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
    )


def _context(request: LiteLLMResponsesRequest) -> Context:
    return Context(
        Route.RESPONSES,
        provider=request.custom_llm_provider,
        model=request.model,
        delivery=Delivery.STREAMING if request.stream else Delivery.COMPLETED,
    )


responses.__doc__ = _python_responses().__doc__
responses.__wrapped__ = _python_responses()  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
aresponses.__doc__ = _python_aresponses().__doc__
aresponses.__wrapped__ = _python_aresponses()  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
