from collections.abc import Coroutine, Mapping, Sequence
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import ValidationError

from litellm.exceptions import BadRequestError
from litellm.rust_bridge import runtime
from litellm.rust_bridge.catalog import Route, RouteContext
from litellm.rust_bridge.dispatch import PublicDispatch
from litellm.rust_bridge.public_call import (
    AsyncFlag,
    NativeCall,
    native_call,
    native_call_hook,
    signature,
)
from litellm.rust_bridge.tokenize.entrypoints import NATIVE_ATOKENIZE, NATIVE_TOKENIZE
from litellm.types.tokenize import TokenizeRepresentation, TokenizeRequest, TokenizeResponse

__all__ = ("atokenize", "tokenize")

_UNSUPPORTED_INPUT_FIELDS: Final = frozenset(
    {"messages", "tools", "image", "images", "chat_template", "add_generation_prompt"}
)


def _bind_request(
    model: str,
    prompt: str,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, object] | None = None,
    include: Sequence[TokenizeRepresentation] | None = None,
    **kwargs: object,  # kwargs-ok: public tokenization accepts provider-specific options
) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "model": model,
            "prompt": prompt,
            "api_key": api_key,
            "api_base": api_base,
            "timeout": timeout,
            "custom_llm_provider": custom_llm_provider,
            "extra_headers": extra_headers,
            "include": include,
            "kwargs": kwargs,
        }
    )


_TOKENIZE: Final = signature(_bind_request)
_FLAG: Final = AsyncFlag("atokenize", sync=_TOKENIZE)


def _public_request(name: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> NativeCall:
    try:
        bound: Final = _bind_request(*args, **kwargs)  # pyright: ignore[reportArgumentType]  # bind before native validation
        supplied: Final = _TOKENIZE.bind(*args, **kwargs).arguments
    except TypeError as error:
        raise TypeError(str(error).replace("_bind_request()", f"{name}()")) from None
    try:
        TokenizeRequest.model_validate(
            {
                name: value
                for name, value in {**bound, **kwargs}.items()
                if name in TokenizeRequest.model_fields and (name != "include" or name in supplied)
            }
        )
        if "output" in kwargs:
            raise ValueError("output is unsupported")
        unsupported_fields: Final = _UNSUPPORTED_INPUT_FIELDS.intersection(kwargs)
        if unsupported_fields:
            raise ValueError(f"{min(unsupported_fields)} is not supported by tokenize")
    except (ValidationError, ValueError) as error:
        raise BadRequestError(
            message=str(error),
            model=str(bound["model"]),
            llm_provider="tokenize",
        ) from error
    return native_call(_TOKENIZE, args, kwargs)


_DISPATCH: Final = PublicDispatch(
    route=Route.TOKENIZE,
    request=lambda args, kwargs: _public_request("tokenize", args, kwargs),
    context=lambda request: RouteContext(Route.TOKENIZE, model=str(request.resolved["model"])),
)
_ADISPATCH: Final = PublicDispatch(
    route=Route.TOKENIZE,
    request=lambda args, kwargs: _public_request("atokenize", args, kwargs),
    context=lambda request: RouteContext(Route.TOKENIZE, model=str(request.resolved["model"])),
)


def tokenize(
    *args: object,
    **kwargs: object,  # kwargs-ok: preserve the public SDK keyword contract
) -> TokenizeResponse | Coroutine[object, object, TokenizeResponse]:
    if _FLAG.requested(kwargs):
        return atokenize(*args, **kwargs)
    return _DISPATCH.run(
        args,
        _FLAG.dropped(kwargs),
        python=runtime.NO_PYTHON,
        binding=NATIVE_TOKENIZE,
        native=native_call_hook,
    )


async def atokenize(
    *args: object,
    **kwargs: object,  # kwargs-ok: public tokenization accepts provider-specific options
) -> TokenizeResponse:
    return await _ADISPATCH.arun(
        args,
        _FLAG.dropped(kwargs),
        python=runtime.NO_PYTHON,
        binding=NATIVE_ATOKENIZE,
        native=native_call_hook,
    )
