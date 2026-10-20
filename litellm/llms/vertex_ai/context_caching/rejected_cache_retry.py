from collections.abc import Awaitable, Callable
from copy import deepcopy
from functools import partial
from typing import Final, TypeVar

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.vertex_ai import RequestBody
from litellm.utils import is_cached_message

from .known_cached_contents import CACHE_REJECTED_STATUS_CODES

Built: Final = TypeVar("Built")
Sent: Final = TypeVar("Sent")


def cache_rebuild(build: Callable[..., Built], transform_params: dict) -> Callable[[], Built] | None:
    """Copies the inputs before the first transform, which pops keys from ``optional_params`` and ``messages``."""
    if not any(map(is_cached_message, transform_params["messages"])):
        return None
    return partial(
        build,
        **{
            **transform_params,
            **deepcopy(
                {"messages": transform_params["messages"], "optional_params": transform_params["optional_params"]}
            ),
        },
    )


def _status_code(error: httpx.HTTPStatusError | BaseLLMException) -> int:
    match error:
        case httpx.HTTPStatusError():
            return error.response.status_code
        case BaseLLMException():
            return error.status_code


def _cache_rejected(body: RequestBody, error: httpx.HTTPStatusError | BaseLLMException) -> bool:
    return body.get("cachedContent") is not None and _status_code(error) in CACHE_REJECTED_STATUS_CODES


async def post_with_fresh_cache_async(
    body: RequestBody,
    rebuild: Callable[[], Awaitable[RequestBody]] | None,
    post: Callable[[RequestBody], Awaitable[Sent]],
) -> tuple[RequestBody, Sent]:
    try:
        return body, await post(body)
    except (httpx.HTTPStatusError, BaseLLMException) as error:
        if rebuild is None or not _cache_rejected(body, error):
            raise
        fresh_body: Final = await rebuild()
        if fresh_body.get("cachedContent") == body.get("cachedContent"):
            raise
    return fresh_body, await post(fresh_body)


def post_with_fresh_cache_sync(
    body: RequestBody,
    rebuild: Callable[[], RequestBody] | None,
    post: Callable[[RequestBody], Sent],
) -> tuple[RequestBody, Sent]:
    try:
        return body, post(body)
    except (httpx.HTTPStatusError, BaseLLMException) as error:
        if rebuild is None or not _cache_rejected(body, error):
            raise
        fresh_body: Final = rebuild()
        if fresh_body.get("cachedContent") == body.get("cachedContent"):
            raise
    return fresh_body, post(fresh_body)
