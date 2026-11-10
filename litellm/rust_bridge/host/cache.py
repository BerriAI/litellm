from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.add_retry_fallback_headers import get_hidden_params_dict

if TYPE_CHECKING:
    from litellm.caching.caching import Cache

_RUST_RESPONSE_HEADER: Final = "x-litellm-rust"
_HEADERS: Final = TypeAdapter(dict[str, object])


def configured_cache() -> Cache | None:
    import litellm

    return litellm.cache


def _python_entry(response: Mapping[str, object]) -> Mapping[str, object]:
    hidden: Final = get_hidden_params_dict(response)
    headers: Final = hidden.get("additional_headers")
    if not isinstance(headers, Mapping) or _RUST_RESPONSE_HEADER not in headers:
        return response
    return {
        **response,
        "_hidden_params": {
            **hidden,
            "additional_headers": {
                name: value
                for name, value in _HEADERS.validate_python(headers).items()
                if name != _RUST_RESPONSE_HEADER
            },
        },
    }


def store_response(
    cache: Cache,
    response: Mapping[str, object],
    /,
    **kwargs: object,  # kwargs-ok: the cache call's own arguments
) -> None:
    from litellm.caching.caching_handler import create_cache_write_task

    create_cache_write_task(
        lambda: cache.async_add_cache(  # pyright: ignore[reportUnknownMemberType]  # Cache.async_add_cache leaves its result untyped
            _python_entry(response),
            **kwargs,  # pyright: ignore[reportArgumentType]  # the cache call's own keyword arguments, as Python passes them
        )
    )
