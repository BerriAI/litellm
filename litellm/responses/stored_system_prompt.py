"""A stored Responses chain keeps the system prompt an earlier turn sent as input, so a
follow-up turn on that chain must not send the same prompt again as `instructions`."""

from collections.abc import Iterator, Mapping
from typing import Final

import httpx
import openai

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping

_SYSTEM_ROLES: Final = frozenset({"system", "developer"})


def _sent(request: Mapping[str, object], key: str) -> object:
    extra_body: Final = as_str_mapping(request.get("extra_body")) or {}
    return extra_body.get(key, request.get(key))


def _continued_response_id(request: Mapping[str, object]) -> str | None:
    instructions: Final = _sent(request, "instructions")
    previous_response_id: Final = _sent(request, "previous_response_id")
    if isinstance(instructions, str) and instructions and isinstance(previous_response_id, str):
        return previous_response_id
    return None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _timeout(value: object) -> float | httpx.Timeout | None:
    return value if isinstance(value, (int, float, httpx.Timeout)) else None


def _stored_system_texts(listed: object) -> Iterator[str]:
    for listed_item in as_object_list((as_str_mapping(listed) or {}).get("data")) or ():
        item = as_str_mapping(listed_item) or {}
        if item.get("role") not in _SYSTEM_ROLES:
            break
        content = item.get("content")
        if isinstance(content, str):
            yield content
            continue
        for listed_part in as_object_list(content) or ():
            part = as_str_mapping(listed_part) or {}
            text = part.get("text")
            if part.get("type") == "input_text" and isinstance(text, str):
                yield text


def _without_instructions(request: Mapping[str, object], listed: object) -> Mapping[str, object]:
    texts: Final = tuple(_stored_system_texts(listed))
    if not texts or _sent(request, "instructions") not in (" ".join(texts), "\n".join(texts)):
        return request
    extra_body: Final = as_str_mapping(request.get("extra_body"))
    return {
        **{key: value for key, value in request.items() if key != "instructions"},
        **(
            {"extra_body": {key: value for key, value in extra_body.items() if key != "instructions"}}
            if extra_body is not None
            else {}
        ),
    }


def without_stored_instructions(request: Mapping[str, object]) -> Mapping[str, object]:
    response_id: Final = _continued_response_id(request)
    if response_id is None:
        return request
    from litellm.responses.main import list_input_items

    try:
        listed: Final[object] = list_input_items(
            response_id=response_id,
            after=None,
            before=None,
            include=None,
            limit=20,
            order="asc",
            extra_headers=None,
            custom_llm_provider=_text(request.get("custom_llm_provider")),
            timeout=_timeout(request.get("timeout")),
            api_key=request.get("api_key"),
            api_base=request.get("api_base"),
            api_version=request.get("api_version"),
            **{"no-log": True},  # the listed stored messages must not reach logging callbacks
        )
    except (openai.OpenAIError, httpx.HTTPError) as error:
        verbose_logger.debug("Keeping instructions, the stored chain could not be read: %s", error)
        return request
    return _without_instructions(request, listed)


async def awithout_stored_instructions(request: Mapping[str, object]) -> Mapping[str, object]:
    response_id: Final = _continued_response_id(request)
    if response_id is None:
        return request
    from litellm.responses.main import alist_input_items

    try:
        listed: Final[object] = await alist_input_items(
            response_id=response_id,
            after=None,
            before=None,
            include=None,
            limit=20,
            order="asc",
            extra_headers=None,
            custom_llm_provider=_text(request.get("custom_llm_provider")),
            timeout=_timeout(request.get("timeout")),
            api_key=request.get("api_key"),
            api_base=request.get("api_base"),
            api_version=request.get("api_version"),
            **{"no-log": True},  # the listed stored messages must not reach logging callbacks
        )
    except (openai.OpenAIError, httpx.HTTPError) as error:
        verbose_logger.debug("Keeping instructions, the stored chain could not be read: %s", error)
        return request
    return _without_instructions(request, listed)
