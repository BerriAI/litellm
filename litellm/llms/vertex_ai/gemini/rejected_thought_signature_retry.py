from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from typing import Final, TypeVar

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm import verbose_logger
from litellm.litellm_core_utils.prompt_templates.factory import get_dummy_thought_signature
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.vertex_ai import ContentType, PartType, RequestBody

from ..context_caching.rejected_cache_retry import status_and_message

Body: Final = TypeVar("Body", bound=Mapping[str, object])
Sent: Final = TypeVar("Sent")

_REJECTED_SIGNATURE_MARKER: Final = "thought signature"
_SIGNATURE_KEYS: Final = frozenset({"thoughtSignature", "thought_signature"})
_ENTRIES: Final = TypeAdapter(list[dict[str, object]])


def names_rejected_thought_signature(status_code: int, message: str) -> bool:
    """A signature only validates on the backend that minted it. AI Studio answers a replayed Vertex signature
    with ``400 Corrupted thought signature.`` and Vertex answers an AI Studio one with ``400 Invalid thought
    signature.``"""
    return status_code == 400 and _REJECTED_SIGNATURE_MARKER in message.lower()


def _is_rejected_signature(error: httpx.HTTPStatusError | BaseLLMException) -> bool:
    return names_rejected_thought_signature(*status_and_message(error))


def _entries(value: object) -> Sequence[Mapping[str, object]]:
    try:
        return _ENTRIES.validate_python(value)
    except ValidationError:
        return []


def _foreign_signatures(body: Mapping[str, object], placeholder: str) -> Iterator[object]:
    for content in _entries(body.get("contents")):
        for part in _entries(content.get("parts")):
            yield from (
                value for key, value in part.items() if key in _SIGNATURE_KEYS and value and value != placeholder
            )


def _foreign_signature_count(body: Mapping[str, object]) -> int:
    return sum(1 for _ in _foreign_signatures(body, get_dummy_thought_signature()))


def _part_with_placeholder(part: PartType, placeholder: str) -> PartType:
    if not part.get("thoughtSignature"):
        return part
    return {**part, "thoughtSignature": placeholder}


def _content_with_placeholders(content: ContentType, placeholder: str) -> ContentType:
    return {**content, "parts": [_part_with_placeholder(part, placeholder) for part in content["parts"]]}


def with_placeholder_thought_signatures(body: RequestBody) -> RequestBody | None:
    """Swaps every thought signature in the history for Google's documented skip-validator placeholder, or
    returns ``None`` when the body carries nothing the backend could have rejected."""
    placeholder: Final = get_dummy_thought_signature()
    if next(_foreign_signatures(body, placeholder), None) is None:
        return None
    return {**body, "contents": [_content_with_placeholders(content, placeholder) for content in body["contents"]]}


def _native_part_with_placeholder(part: Mapping[str, object], placeholder: str) -> Mapping[str, object]:
    return {key: placeholder if key in _SIGNATURE_KEYS and value else value for key, value in part.items()}


def _native_content_with_placeholders(content: Mapping[str, object], placeholder: str) -> Mapping[str, object]:
    parts: Final = _entries(content.get("parts"))
    if not parts:
        return content
    return {**content, "parts": [_native_part_with_placeholder(part, placeholder) for part in parts]}


def with_placeholder_native_thought_signatures(body: Mapping[str, object]) -> Mapping[str, object] | None:
    """The same swap on a ``generateContent`` body as the client sent it, whose parts may spell the key
    ``thoughtSignature`` or ``thought_signature``."""
    placeholder: Final = get_dummy_thought_signature()
    if next(_foreign_signatures(body, placeholder), None) is None:
        return None
    return {
        **body,
        "contents": [
            _native_content_with_placeholders(content, placeholder) for content in _entries(body.get("contents"))
        ],
    }


async def _inline_history_async(body: Body, uncached: Callable[[], Awaitable[Body]] | None) -> Body | None:
    if uncached is None or body.get("cachedContent") is None:
        return None
    inline: Final = await uncached()
    return inline if _foreign_signature_count(inline) > _foreign_signature_count(body) else None


def _inline_history_sync(body: Body, uncached: Callable[[], Body] | None) -> Body | None:
    if uncached is None or body.get("cachedContent") is None:
        return None
    inline: Final = uncached()
    return inline if _foreign_signature_count(inline) > _foreign_signature_count(body) else None


def _resend_body(signed: Body, placeholders: Callable[[Body], Body | None]) -> Body | None:
    resend_body: Final = placeholders(signed)
    if resend_body is not None:
        verbose_logger.warning(
            "Gemini rejected a replayed thought signature, resending with the skip_thought_signature_validator"
            " placeholder in place of the replayed signatures"
        )
    return resend_body


async def build_with_accepted_thought_signatures_async(
    build: Callable[[], Awaitable[Body]],
    placeholders: Callable[[Body], Body | None],
    uncached: Callable[[], Awaitable[Body]] | None,
) -> Body:
    """Vertex validates the signatures in a cache's contents when it creates the cache, so a rejected one there
    sends the history inline with the placeholder instead."""
    try:
        return await build()
    except BaseLLMException as error:
        if uncached is None or not _is_rejected_signature(error):
            raise
        inline_body: Final = _resend_body(await uncached(), placeholders)
        if inline_body is None:
            raise
    return inline_body


def build_with_accepted_thought_signatures_sync(
    build: Callable[[], Body],
    placeholders: Callable[[Body], Body | None],
    uncached: Callable[[], Body] | None,
) -> Body:
    try:
        return build()
    except BaseLLMException as error:
        if uncached is None or not _is_rejected_signature(error):
            raise
        inline_body: Final = _resend_body(uncached(), placeholders)
        if inline_body is None:
            raise
    return inline_body


async def _post_inline_async(
    body: Body,
    placeholders: Callable[[Body], Body | None],
    uncached: Callable[[], Awaitable[Body]] | None,
    post: Callable[[Body], Awaitable[Sent]],
    log_request: Callable[[Body], None],
    rejection: httpx.HTTPStatusError | BaseLLMException,
) -> tuple[Body, Sent]:
    inline: Final = await _inline_history_async(body, uncached)
    resend_body: Final = None if inline is None else _resend_body(inline, placeholders)
    if resend_body is None:
        raise rejection
    log_request(resend_body)
    return resend_body, await post(resend_body)


def _post_inline_sync(
    body: Body,
    placeholders: Callable[[Body], Body | None],
    uncached: Callable[[], Body] | None,
    post: Callable[[Body], Sent],
    log_request: Callable[[Body], None],
    rejection: httpx.HTTPStatusError | BaseLLMException,
) -> tuple[Body, Sent]:
    inline: Final = _inline_history_sync(body, uncached)
    resend_body: Final = None if inline is None else _resend_body(inline, placeholders)
    if resend_body is None:
        raise rejection
    log_request(resend_body)
    return resend_body, post(resend_body)


async def post_with_accepted_thought_signatures_async(
    body: Body,
    placeholders: Callable[[Body], Body | None],
    uncached: Callable[[], Awaitable[Body]] | None,
    post: Callable[[Body], Awaitable[Sent]],
    log_request: Callable[[Body], None],
) -> tuple[Body, Sent]:
    """Innermost retry around the raw post, so a cache rebuild (which re-transforms the messages and brings
    the rejected signatures back) is itself wrapped by it. The resend keeps the cache and swaps the signatures
    the body carries, and sends the history inline only when that is rejected too and the cache holds one.
    Returns the body the backend accepted."""
    try:
        return body, await post(body)
    except (httpx.HTTPStatusError, BaseLLMException) as error:
        if not _is_rejected_signature(error):
            raise
        resend_body: Final = _resend_body(body, placeholders)
        if resend_body is None:
            return await _post_inline_async(body, placeholders, uncached, post, log_request, error)
    log_request(resend_body)
    try:
        return resend_body, await post(resend_body)
    except (httpx.HTTPStatusError, BaseLLMException) as error:
        if not _is_rejected_signature(error):
            raise
        return await _post_inline_async(body, placeholders, uncached, post, log_request, error)


def post_with_accepted_thought_signatures_sync(
    body: Body,
    placeholders: Callable[[Body], Body | None],
    uncached: Callable[[], Body] | None,
    post: Callable[[Body], Sent],
    log_request: Callable[[Body], None],
) -> tuple[Body, Sent]:
    try:
        return body, post(body)
    except (httpx.HTTPStatusError, BaseLLMException) as error:
        if not _is_rejected_signature(error):
            raise
        resend_body: Final = _resend_body(body, placeholders)
        if resend_body is None:
            return _post_inline_sync(body, placeholders, uncached, post, log_request, error)
    log_request(resend_body)
    try:
        return resend_body, post(resend_body)
    except (httpx.HTTPStatusError, BaseLLMException) as error:
        if not _is_rejected_signature(error):
            raise
        return _post_inline_sync(body, placeholders, uncached, post, log_request, error)
