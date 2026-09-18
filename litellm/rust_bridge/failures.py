"""Map a native failure onto LiteLLM's public exception contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol, cast  # noqa: TID251  # adapts the public exception mapper

from pydantic import TypeAdapter

import litellm
from litellm.rust_bridge.bindings import native_exception_types


@dataclass(frozen=True, slots=True)
class Declined:
    """The native route rejected the call before any provider I/O."""

    message: str
    rejection: str


@dataclass(frozen=True, slots=True)
class Upstream:
    """The provider was called and did not succeed; status 0 means no response arrived."""

    status: int
    body: str
    headers: tuple[tuple[str, str], ...]


_DECLINED_ARGS: Final = TypeAdapter(tuple[str, str])
_UPSTREAM_ARGS: Final = TypeAdapter(tuple[int, str, tuple[tuple[str, str], ...]])


def native_failure(error: BaseException) -> Declined | Upstream | None:
    """Decode a native exception's args once, for every consumer of the bridge."""
    exceptions: Final = native_exception_types()
    if exceptions is None:
        return None
    declined, upstream = exceptions
    if isinstance(error, declined):
        message, rejection = _DECLINED_ARGS.validate_python(error.args)
        return Declined(message, rejection)
    if isinstance(error, upstream):
        status, body, headers = _UPSTREAM_ARGS.validate_python(error.args)
        return Upstream(status, body, headers)
    return None


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
