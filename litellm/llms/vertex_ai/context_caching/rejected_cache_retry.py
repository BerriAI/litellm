from collections.abc import Awaitable, Callable
from copy import deepcopy
from functools import partial
from typing import Final, TypeVar

import httpx

from litellm.litellm_core_utils.core_helpers import independent_snapshot
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.vertex_ai import RequestBody
from litellm.utils import is_cached_message

from .known_cached_contents import names_rejected_cache

Built: Final = TypeVar("Built")
Sent: Final = TypeVar("Sent")


def cache_rebuild(
    build: Callable[..., Built], messages: list[AllMessageValues], optional_params: dict
) -> Callable[[], Built] | None:
    """Copies the inputs before the first transform, which pops keys from ``optional_params`` and ``messages``."""
    if not any(map(is_cached_message, messages)):
        return None
    return partial(build, messages=deepcopy(messages), optional_params=independent_snapshot(optional_params))


def _status_and_message(error: httpx.HTTPStatusError | BaseLLMException) -> tuple[int, str]:
    match error:
        case httpx.HTTPStatusError():
            return error.response.status_code, error.response.text
        case BaseLLMException():
            return error.status_code, error.message


def _cache_rejected(body: RequestBody, error: httpx.HTTPStatusError | BaseLLMException) -> bool:
    return body.get("cachedContent") is not None and names_rejected_cache(*_status_and_message(error))


async def post_with_fresh_cache_async(
    body: RequestBody,
    rebuild: Callable[[], Awaitable[RequestBody]] | None,
    post: Callable[[RequestBody], Awaitable[Sent]],
    log_request: Callable[[RequestBody], None],
) -> tuple[RequestBody, Sent]:
    try:
        return body, await post(body)
    except (httpx.HTTPStatusError, BaseLLMException) as error:
        if rebuild is None or not _cache_rejected(body, error):
            raise
        fresh_body: Final = await rebuild()
        log_request(fresh_body)
        if fresh_body.get("cachedContent") == body.get("cachedContent"):
            raise
    return fresh_body, await post(fresh_body)


def post_with_fresh_cache_sync(
    body: RequestBody,
    rebuild: Callable[[], RequestBody] | None,
    post: Callable[[RequestBody], Sent],
    log_request: Callable[[RequestBody], None],
) -> tuple[RequestBody, Sent]:
    try:
        return body, post(body)
    except (httpx.HTTPStatusError, BaseLLMException) as error:
        if rebuild is None or not _cache_rejected(body, error):
            raise
        fresh_body: Final = rebuild()
        log_request(fresh_body)
        if fresh_body.get("cachedContent") == body.get("cachedContent"):
            raise
    return fresh_body, post(fresh_body)
