"""Map a native failure onto LiteLLM's public exception contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Protocol, cast  # noqa: TID251  # adapts the public exception mapper

import httpx
import openai
from pydantic import TypeAdapter, ValidationError

import litellm

_UPSTREAM_ARGS: Final = TypeAdapter(tuple[int, str])
_UPSTREAM_HEADERS: Final = TypeAdapter(list[tuple[str, str]])


class UpstreamFailure(Exception):
    def __init__(self, response: httpx.Response, cause: Exception) -> None:
        super().__init__(str(cause))
        self.message: Final = str(cause)
        self.response: Final = response
        self.status_code: Final = response.status_code
        self.__cause__ = cause


def _upstream_failure(error: Exception, api_base: str | None) -> Exception:
    try:
        status, body = _UPSTREAM_ARGS.validate_python(error.args)
        headers: Final = _UPSTREAM_HEADERS.validate_python(getattr(error, "headers", None))
    except ValidationError:
        return error
    http_request: Final = httpx.Request("POST", api_base or "https://docs.litellm.ai/docs")
    return UpstreamFailure(
        httpx.Response(status, content=body.encode(), headers=headers, request=http_request),
        error,
    )


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
            completion_kwargs=dict(kwargs),  # mutable-ok: exception mapper requires owned kwargs
            extra_kwargs=dict(kwargs),  # mutable-ok: exception mapper requires owned kwargs
        )
    except Exception as public_error:
        public_error.__context__ = error
        return public_error


def map_native_failure(
    error: Exception, model: str, request_provider: str, kwargs: Mapping[str, object], api_base: str | None = None
) -> Exception:
    """`map_failure`, reading a native `(status, body)` provider failure as the HTTP response it was."""
    original: Final = _upstream_failure(error, api_base)
    public_error: Final = map_failure(original, model, request_provider, kwargs)
    if isinstance(original, UpstreamFailure) and public_error.__context__ is original:
        public_error.__context__ = error
        if isinstance(public_error, openai.APIStatusError):
            public_error.response = original.response
            public_error.status_code = original.status_code
    return public_error
