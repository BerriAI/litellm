"""Map a native failure onto LiteLLM's public exception contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Protocol, cast  # noqa: TID251  # adapts the public exception mapper

import litellm


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
