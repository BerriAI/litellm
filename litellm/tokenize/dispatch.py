from collections.abc import Coroutine, Mapping
from types import MappingProxyType
from typing import Final

import httpx

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
from litellm.types.tokenize import TokenizeResponse

__all__ = ("atokenize", "tokenize")


def _bind_request(
    model: str,
    prompt: str,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, object] | None = None,
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
            "kwargs": kwargs,
        }
    )


_TOKENIZE: Final = signature(_bind_request)
_FLAG: Final = AsyncFlag("atokenize", sync=_TOKENIZE)


def _public_request(name: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> NativeCall:
    try:
        _bind_request(*args, **kwargs)  # pyright: ignore[reportArgumentType]  # bind before native validation
    except TypeError as error:
        raise TypeError(str(error).replace("_bind_request()", f"{name}()")) from None
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
    *args: object, **kwargs: object
) -> TokenizeResponse | Coroutine[object, object, TokenizeResponse]:  # kwargs-ok: public SDK signature
    if _FLAG.requested(kwargs):
        return atokenize(**_FLAG.keywords(args, kwargs))
    return _DISPATCH.run(
        args,
        _FLAG.dropped(kwargs),
        python=runtime.NO_PYTHON,
        binding=NATIVE_TOKENIZE,
        native=native_call_hook,
    )


async def atokenize(*args: object, **kwargs: object) -> TokenizeResponse:  # kwargs-ok: public SDK signature
    return await _ADISPATCH.arun(
        args,
        _FLAG.dropped(kwargs),
        python=runtime.NO_PYTHON,
        binding=NATIVE_ATOKENIZE,
        native=native_call_hook,
    )
