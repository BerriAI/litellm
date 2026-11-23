from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from typing import Final, TypeVar

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm import verbose_logger
from litellm.litellm_core_utils.prompt_templates.factory import get_dummy_thought_signature
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.vertex_ai import ContentType, PartType, RequestBody

Sent: Final = TypeVar("Sent")

_REJECTED_SIGNATURE_MARKER: Final = "thought signature"
_SIGNATURE_KEYS: Final = frozenset({"thoughtSignature", "thought_signature"})
_ENTRIES: Final = TypeAdapter(list[dict[str, object]])


def names_rejected_thought_signature(status_code: int, message: str) -> bool:
    """A signature only validates on the backend that minted it. AI Studio answers a replayed Vertex signature
    with ``400 Corrupted thought signature.`` and Vertex answers an AI Studio one with ``400 Invalid thought
    signature.``"""
    return status_code == 400 and _REJECTED_SIGNATURE_MARKER in message.lower()


def _status_and_message(error: httpx.HTTPStatusError | BaseLLMException) -> tuple[int, str]:
    match error:
        case httpx.HTTPStatusError():
            return error.response.status_code, error.response.text
        case BaseLLMException():
            return error.status_code, error.message


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


def _resend_body(body: RequestBody, error: httpx.HTTPStatusError | BaseLLMException) -> RequestBody | None:
    if not names_rejected_thought_signature(*_status_and_message(error)):
        return None
    resend_body: Final = with_placeholder_thought_signatures(body)
    if resend_body is not None:
        verbose_logger.warning(
            "Gemini rejected a replayed thought signature, resending with the skip_thought_signature_validator"
            " placeholder in place of the replayed signatures"
        )
    return resend_body


async def post_with_accepted_thought_signatures_async(
    body: RequestBody,
    post: Callable[[RequestBody], Awaitable[Sent]],
    log_request: Callable[[RequestBody], None],
) -> tuple[RequestBody, Sent]:
    try:
        return body, await post(body)
    except (httpx.HTTPStatusError, BaseLLMException) as error:
        resend_body: Final = _resend_body(body, error)
        if resend_body is None:
            raise
    log_request(resend_body)
    return resend_body, await post(resend_body)


def post_with_accepted_thought_signatures_sync(
    body: RequestBody,
    post: Callable[[RequestBody], Sent],
    log_request: Callable[[RequestBody], None],
) -> tuple[RequestBody, Sent]:
    try:
        return body, post(body)
    except (httpx.HTTPStatusError, BaseLLMException) as error:
        resend_body: Final = _resend_body(body, error)
        if resend_body is None:
            raise
    log_request(resend_body)
    return resend_body, post(resend_body)
