"""Read the report a native failure carries and build the public exception for it.

The Rust side raises one exception, ``RustFailure``, whose only argument is a report: the
``stage`` the call failed at, the ``kind`` of failure and a message. Rust settles every failure
before it crosses: a bare ``RustFailure`` means the attempt was abandoned for Python to serve the
call, and anything else is the public LiteLLM exception that `public_exception` builds here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Final, Literal, Protocol, cast  # noqa: TID251  # adapts the public exception mapper

import httpx
import openai
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

import litellm
from litellm.rust_bridge.bindings import native_failure_type

Stage = Literal["prepare", "send", "upstream", "receive", "post_call"]


class UpstreamKind(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["upstream"]
    status: int
    headers: tuple[tuple[str, str], ...]
    body: str
    url: str | None


class PlainKind(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["request", "unsupported", "auth", "timeout", "connection", "response", "internal"]


class NativeFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Stage
    kind: Annotated[UpstreamKind | PlainKind, Field(discriminator="kind")]
    message: str


_REPORT: Final = TypeAdapter(NativeFailure)
_FALLBACK_URL: Final = "https://docs.litellm.ai/docs"


def report(error: BaseException) -> NativeFailure | None:
    """The report a bare ``RustFailure`` carries as its one argument; nothing for any other exception."""
    native: Final = native_failure_type()
    if native is None or not isinstance(error, native):
        return None
    try:
        return _REPORT.validate_python(error.args[0] if error.args else None)
    except ValidationError:
        return None


class UpstreamFailure(Exception):
    def __init__(self, response: httpx.Response, cause: Exception) -> None:
        super().__init__(str(cause))
        self.message: Final = str(cause)
        self.response: Final = response
        self.status_code: Final = response.status_code
        self.__cause__ = cause


class ExceptionMapper(Protocol):
    def __call__(
        self,
        *,
        model: str,
        custom_llm_provider: str | None,
        original_exception: Exception,
        completion_kwargs: dict[str, object],  # mutable-ok: the legacy public exception mapper mutates its kwargs
        extra_kwargs: dict[str, object],  # mutable-ok: the legacy public exception mapper mutates its kwargs
    ) -> Exception: ...


def map_failure(error: Exception, model: str, request_provider: str, kwargs: Mapping[str, object]) -> Exception:
    mapper: Final = cast(  # cast-ok: bounded adapter for the legacy public exception mapper
        ExceptionMapper, litellm.exception_type
    )
    try:
        return mapper(
            model=model.removeprefix(f"{request_provider}/"),
            custom_llm_provider=request_provider,
            original_exception=error,
            completion_kwargs=dict(kwargs),
            extra_kwargs=dict(kwargs),
        )
    except Exception as public_error:
        public_error.__context__ = error
        return public_error


def _upstream_exception(
    upstream: UpstreamKind, error: Exception, model: str, provider: str, kwargs: Mapping[str, object]
) -> Exception:
    api_base: Final = kwargs.get("api_base") or kwargs.get("base_url")
    url: Final = upstream.url or (api_base if isinstance(api_base, str) else None) or _FALLBACK_URL
    original: Final = UpstreamFailure(
        httpx.Response(
            upstream.status,
            content=upstream.body.encode(),
            headers=list(upstream.headers),
            request=httpx.Request("POST", url),
        ),
        error,
    )
    public_error: Final = map_failure(original, model, provider, kwargs)
    if public_error.__context__ is original:
        public_error.__context__ = error
        if isinstance(public_error, openai.APIStatusError):
            public_error.response = original.response
            public_error.status_code = original.status_code
    return public_error


def _plain_exception(kind: PlainKind, message: str, model: str, provider: str) -> Exception:
    bare_model: Final = model.removeprefix(f"{provider}/")
    match kind.kind:
        case "request":
            return litellm.BadRequestError(message=message, model=bare_model, llm_provider=provider)
        case "unsupported":
            return litellm.UnsupportedParamsError(message=message, model=bare_model, llm_provider=provider)
        case "auth":
            return litellm.AuthenticationError(message=message, model=bare_model, llm_provider=provider)
        case "timeout":
            return litellm.Timeout(message=message, model=bare_model, llm_provider=provider)
        case "connection":
            return litellm.APIConnectionError(message=message, model=bare_model, llm_provider=provider)
        case "response" | "internal":
            return litellm.APIError(status_code=500, message=message, model=bare_model, llm_provider=provider)


def _provider(model: str, custom_llm_provider: str | None) -> str:
    if custom_llm_provider:
        return custom_llm_provider
    try:
        return litellm.get_llm_provider(model=model)[1]
    except litellm.BadRequestError:
        return model.partition("/")[0] if "/" in model else ""


def public_exception(error: BaseException, request: Mapping[str, object], provider: str | None) -> BaseException:
    """The public LiteLLM exception for a bare native failure; anything else comes back unchanged.

    Called from the Rust side when it settles a failure, so callbacks and the caller see the same
    exception, and from the runtime for native entrypoints that raise the bare ``RustFailure``."""
    failure: Final = report(error)
    if failure is None or not isinstance(error, Exception):
        return error
    model: Final = str(request.get("model") or "")
    custom_llm_provider: Final = request.get("custom_llm_provider")
    llm_provider: Final = provider or _provider(
        model, custom_llm_provider if isinstance(custom_llm_provider, str) else None
    )
    match failure.kind:
        case UpstreamKind() as upstream:
            public: Exception = _upstream_exception(upstream, error, model, llm_provider, request)
        case PlainKind() as plain:
            public = _plain_exception(plain, failure.message, model, llm_provider)
    public.__cause__ = error
    return public
