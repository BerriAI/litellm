"""A stored Responses chain keeps the system prompt an earlier turn sent as input, so a
follow-up turn on that chain must not send the same prompt again as `instructions`. A prompt too
long for `instructions` goes in `input` instead."""

from collections.abc import Iterator, Mapping
from typing import Final

import httpx
import openai

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping

_SYSTEM_ROLES: Final = frozenset({"system", "developer"})
# OpenAI and Azure answer longer `instructions` with 400 string_above_max_length, counting code points (2026-11-19)
RESPONSES_INSTRUCTIONS_MAX_LENGTH: Final = 1_048_576


def _sent(request: Mapping[str, object], key: str) -> object:
    extra_body: Final = as_str_mapping(request.get("extra_body")) or {}
    return extra_body.get(key, request.get(key))


def _continued_response_id(request: Mapping[str, object]) -> str | None:
    instructions: Final = _sent(request, "instructions")
    previous_response_id: Final = _sent(request, "previous_response_id")
    if not isinstance(instructions, str) or not instructions or not isinstance(previous_response_id, str):
        return None
    if _sent(request, "truncation") == "auto" and len(instructions) <= RESPONSES_INSTRUCTIONS_MAX_LENGTH:
        # truncation may cut the stored system message, so a prompt that fits stays in `instructions`
        return None
    return _upstream_response_id(previous_response_id, request.get("custom_llm_provider"))


def _upstream_response_id(response_id: str, provider: object) -> str | None:
    """The provider's own id, or None when the id names a provider other than the request's.

    A LiteLLM-encoded id carries a provider that the listing call would follow, so the caller's id must not
    choose where the request's credentials are sent."""
    from litellm.responses.utils import ResponsesAPIRequestUtils

    decoded: Final = ResponsesAPIRequestUtils.decode_responses_api_response_id(response_id)
    embedded_provider: Final = decoded.get("custom_llm_provider")
    upstream_id: Final = decoded.get("response_id")
    if embedded_provider is None:
        return response_id
    if embedded_provider != provider or not upstream_id:
        return None
    nested: Final = ResponsesAPIRequestUtils.decode_responses_api_response_id(upstream_id)
    return upstream_id if nested.get("custom_llm_provider") is None else None


_FORWARDED_KEYS: Final = (
    "api_key",
    "ssl_verify",
    "api_base",
    "api_version",
    "azure_ad_token",
    "azure_ad_token_provider",
    "tenant_id",
    "client_id",
    "client_secret",
    "azure_username",
    "azure_password",
    "azure_scope",
)


def _lookup_kwargs(request: Mapping[str, object]) -> Mapping[str, object]:
    return {
        **{key: request[key] for key in _FORWARDED_KEYS if request.get(key) is not None},
        "no-log": True,  # the listed stored messages must not reach logging callbacks
    }


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


def _instructions_beyond(instructions: object, stored: str) -> str | None:
    """What `instructions` adds to the stored system text: "" for nothing, None when it does not start with it."""
    if not isinstance(instructions, str) or not instructions.startswith(stored):
        return None
    if instructions == stored:
        return ""
    return instructions[len(stored) + 1 :] if instructions[len(stored)] == "\n" else None


def _without_instructions(request: Mapping[str, object], listed: object) -> Mapping[str, object]:
    texts: Final = tuple(_stored_system_texts(listed))
    instructions: Final = _sent(request, "instructions")
    remaining: Final = next(
        (
            beyond
            for beyond in (_instructions_beyond(instructions, joiner.join(texts)) for joiner in (" ", "\n"))
            if beyond is not None
        ),
        None,
    )
    if not texts or remaining is None:
        return request
    # a caller's own `instructions` follow the system text, and the first turn sent them without it
    return _with_instructions(request, {"instructions": remaining} if remaining else {})


def _with_instructions(request: Mapping[str, object], kept: Mapping[str, str]) -> Mapping[str, object]:
    """`request` with `kept` in place of `instructions`, wherever the request carried them."""
    extra_body: Final = as_str_mapping(request.get("extra_body"))
    return {
        **{key: value for key, value in request.items() if key != "instructions"},
        **(kept if "instructions" in request else {}),
        **(
            {
                "extra_body": {
                    **{key: value for key, value in extra_body.items() if key != "instructions"},
                    **(kept if "instructions" in extra_body else {}),
                }
            }
            if extra_body is not None
            else {}
        ),
    }


def _with_long_instructions_in_input(request: Mapping[str, object]) -> Mapping[str, object]:
    instructions: Final = _sent(request, "instructions")
    input_items: Final = as_object_list(request.get("input"))
    if (
        not isinstance(instructions, str)
        or len(instructions) <= RESPONSES_INSTRUCTIONS_MAX_LENGTH
        or input_items is None
    ):
        return request
    system_item: Final = {
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": instructions}],
    }
    return {**_with_instructions(request, {}), "input": [system_item, *input_items]}


def without_stored_instructions(request: Mapping[str, object]) -> Mapping[str, object]:
    return _with_long_instructions_in_input(_without_stored_instructions(request))


async def awithout_stored_instructions(request: Mapping[str, object]) -> Mapping[str, object]:
    return _with_long_instructions_in_input(await _awithout_stored_instructions(request))


def _without_stored_instructions(request: Mapping[str, object]) -> Mapping[str, object]:
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
            extra_headers=dict(as_str_mapping(request.get("extra_headers")) or {}) or None,
            custom_llm_provider=_text(request.get("custom_llm_provider")),
            timeout=_timeout(request.get("timeout")),
            **_lookup_kwargs(request),
        )
    except (openai.OpenAIError, httpx.HTTPError) as error:
        verbose_logger.debug("Keeping instructions, the stored chain could not be read: %s", error)
        return request
    return _without_instructions(request, listed)


async def _awithout_stored_instructions(request: Mapping[str, object]) -> Mapping[str, object]:
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
            extra_headers=dict(as_str_mapping(request.get("extra_headers")) or {}) or None,
            custom_llm_provider=_text(request.get("custom_llm_provider")),
            timeout=_timeout(request.get("timeout")),
            **_lookup_kwargs(request),
        )
    except (openai.OpenAIError, httpx.HTTPError) as error:
        verbose_logger.debug("Keeping instructions, the stored chain could not be read: %s", error)
        return request
    return _without_instructions(request, listed)
