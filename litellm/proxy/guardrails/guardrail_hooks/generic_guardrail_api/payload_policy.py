"""What the Generic Guardrail API endpoint is sent, and what it may write back.

Shaping is lossy, so every shaped payload carries a ``PayloadLoss``. Caller content the
guardrail did not see in full is never replaced by the guardrail's response.
"""

import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import accumulate, chain
from types import MappingProxyType
from typing import Final, Literal

import regex
from pydantic import JsonValue, PositiveInt, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    GENERIC_GUARDRAIL_IMAGE_OMITTED_PLACEHOLDER,
    GENERIC_GUARDRAIL_UNAPPLIABLE_REWRITE,
    GenericGuardrailUnappliableRewrite,
)
from litellm.llms.base_llm.guardrail_translation.utils import (
    message_slot_texts,
    message_with_slot_texts,
    unappliable_request_rewrite,
)
from litellm.proxy.guardrails._content_utils import image_part_url, map_messages_image_urls
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.config_parsing import config_strings
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
    GenericGuardrailAPIRequest,
    GenericGuardrailAPIResponse,
    GuardrailToolParam,
    structured_messages_from_json,
)

PROTECTED_PAYLOAD_FIELDS: Final = frozenset({"input_type", "litellm_call_id"})

_BOOL: Final = TypeAdapter(bool)
_PARTS: Final[TypeAdapter[tuple[object, ...]]] = TypeAdapter(tuple[object, ...])
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_POSITIVE_INT: Final[TypeAdapter[int]] = TypeAdapter(PositiveInt)

MAX_STRIP_SUBSTITUTIONS: Final = 64

MAX_STRIP_CALL_CHARS: Final = 100_000

STRIP_TIMEOUT_SECONDS: Final = 0.1


@dataclass(frozen=True, slots=True)
class PayloadPolicy:
    send_images: bool = True
    exclude_fields: frozenset[str] = frozenset()
    max_messages: int | None = None
    max_text_chars: int | None = None
    strip_patterns: tuple[regex.Pattern[str], ...] = ()

    @property
    def omitted_fields(self) -> frozenset[str]:
        return self.exclude_fields if self.send_images else self.exclude_fields | frozenset(("images",))

    @property
    def shapes_messages(self) -> bool:
        return not self.send_images

    @property
    def shapes_text(self) -> bool:
        return self.max_text_chars is not None or bool(self.strip_patterns)

    @property
    def lossy_options(self) -> tuple[str, ...]:
        options: Final = (
            ("send_images=False", not self.send_images),
            (f"exclude_payload_fields={sorted(self.exclude_fields)}", bool(self.exclude_fields)),
            (f"max_messages={self.max_messages}", self.max_messages is not None),
            (f"max_text_chars={self.max_text_chars}", self.max_text_chars is not None),
            ("strip_patterns", bool(self.strip_patterns)),
        )
        return tuple(option for option, is_set in options if is_set)

    @property
    def is_lossy(self) -> bool:
        return bool(self.lossy_options)


@dataclass(frozen=True, slots=True)
class PayloadLoss:
    altered_message_indices: frozenset[int] = frozenset()
    texts_omitted: bool = False
    messages_omitted: bool = False
    images_omitted: bool = False
    tools_omitted: bool = False
    text_shaped: bool = False


@dataclass(frozen=True, slots=True)
class ShapedPayload:
    body: dict[str, JsonValue]  # mutable-ok: the HTTP client takes the POST body as a dict
    sent_messages: Sequence[AllMessageValues] | None
    loss: PayloadLoss


@dataclass(frozen=True, slots=True)
class AcceptedRewrites:
    texts: list[str] | None  # mutable-ok: guardrail inputs take a list
    images: list[str] | None  # mutable-ok: guardrail inputs take a list
    tools: list[GuardrailToolParam] | None  # mutable-ok: guardrail inputs take a list
    rows: Sequence[AllMessageValues] | None
    refused_any: bool


def resolve_payload_policy(
    *,
    send_images: object,
    exclude_payload_fields: object,
    max_messages: object,
    max_text_chars: object,
    strip_patterns: object,
    guardrail_name: str | None,
) -> PayloadPolicy:
    policy: Final = PayloadPolicy(
        send_images=_send_images(send_images),
        exclude_fields=_resolve_exclude_fields(
            config_strings(
                exclude_payload_fields, option_name="exclude_payload_fields", fallback="Every field is sent"
            ),
            guardrail_name=guardrail_name,
        ),
        max_messages=_positive_int(max_messages, option_name="max_messages", fallback="Every message is sent"),
        max_text_chars=_positive_int(max_text_chars, option_name="max_text_chars", fallback="Texts are sent in full"),
        strip_patterns=_compile_strip_patterns(strip_patterns),
    )
    if policy.is_lossy:
        verbose_proxy_logger.warning(
            "Generic Guardrail API (%s): %s keep part of the request from the guardrail, so it can only enforce "
            "on what it is sent, and it cannot rewrite what it did not see.",
            guardrail_name,
            ", ".join(policy.lossy_options),
        )
    return policy


def _parsed_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return _POSITIVE_INT.validate_python(value)
    except ValidationError:
        return None


def _positive_int(value: object, *, option_name: str, fallback: str) -> int | None:
    if value is None:
        return None
    parsed: Final = _parsed_positive_int(value)
    if parsed is None:
        verbose_proxy_logger.warning(
            "Ignoring %s=%r, expected a whole number of at least 1. %s", option_name, value, fallback
        )
    return parsed


def _compiled_pattern(pattern: str) -> regex.Pattern[str] | None:
    try:
        return regex.compile(pattern)
    except (regex.error, RecursionError) as error:
        verbose_proxy_logger.warning(
            "Ignoring strip_patterns entry %r, it is not a valid regex: %s. The other patterns still apply",
            pattern,
            error,
        )
        return None


def _compile_strip_patterns(raw: object) -> tuple[regex.Pattern[str], ...]:
    compiled: Final = tuple(
        _compiled_pattern(pattern)
        for pattern in config_strings(raw, option_name="strip_patterns", fallback="Nothing is stripped")
    )
    return tuple(pattern for pattern in compiled if pattern is not None)


def _send_images(value: object) -> bool:
    if value is None:
        return True
    try:
        return _BOOL.validate_python(value)
    except ValidationError:
        verbose_proxy_logger.warning("Ignoring send_images=%r, expected a bool. Images are sent", value)
        return True


def _resolve_exclude_fields(raw: Sequence[str], *, guardrail_name: str | None) -> frozenset[str]:
    known: Final = frozenset(GenericGuardrailAPIRequest.model_fields)
    unknown: Final = tuple(field for field in raw if field not in known)
    if unknown:
        verbose_proxy_logger.warning(
            "Generic Guardrail API (%s): ignoring unknown exclude_payload_fields %s. Known fields: %s",
            guardrail_name,
            unknown,
            sorted(known),
        )
    protected: Final = tuple(field for field in raw if field in PROTECTED_PAYLOAD_FIELDS)
    if protected:
        verbose_proxy_logger.warning(
            "Generic Guardrail API (%s): exclude_payload_fields cannot drop %s, the guardrail needs them to "
            "interpret the payload; they are still sent.",
            guardrail_name,
            protected,
        )
    excludable: Final = known - PROTECTED_PAYLOAD_FIELDS
    return frozenset(field for field in raw if field in excludable)


def _omit_image(_url: str) -> str:
    return GENERIC_GUARDRAIL_IMAGE_OMITTED_PLACEHOLDER


def _content(row: Mapping[str, object]) -> object:
    return row.get("content")


def _part_list(value: object) -> tuple[object, ...] | None:
    return _PARTS.validate_python(value) if isinstance(value, (list, tuple)) else None


def _holds_placeholder(part: object) -> bool:
    return image_part_url(part) == GENERIC_GUARDRAIL_IMAGE_OMITTED_PLACEHOLDER


def _altered_message_indices(unshaped: JsonValue, sent: JsonValue) -> frozenset[int]:
    if not isinstance(unshaped, list) or not isinstance(sent, list):
        return frozenset()
    return frozenset(index for index, (row, sent_row) in enumerate(zip(unshaped, sent, strict=True)) if row != sent_row)


def _string_list(value: JsonValue) -> tuple[str, ...]:
    return tuple(item for item in value if isinstance(item, str)) if isinstance(value, list) else ()


def _windowed_messages(messages: JsonValue, max_messages: int | None) -> JsonValue:
    if max_messages is None or not isinstance(messages, list) or len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]


def _strip(text: str, patterns: tuple[regex.Pattern[str], ...], deadline: float) -> str | None:
    def strip_one(acc: str | None, pattern: regex.Pattern[str]) -> str | None:
        remaining: Final = deadline - time.monotonic()
        if acc is None or remaining <= 0:
            return None
        try:
            return pattern.sub("", acc, count=MAX_STRIP_SUBSTITUTIONS, timeout=remaining)
        except TimeoutError:
            return None

    return reduce(strip_one, patterns, text)


def _stripped_fragments(
    fragments: Iterable[str], policy: PayloadPolicy, guardrail_name: str | None
) -> Mapping[str, str]:
    if not policy.strip_patterns:
        return MappingProxyType({})
    distinct: Final = tuple(dict.fromkeys(fragments))
    spent: Final = tuple(accumulate(len(fragment) for fragment in distinct))
    within_budget: Final = tuple(fragment for fragment, total in zip(distinct, spent) if total <= MAX_STRIP_CALL_CHARS)
    deadline: Final = time.monotonic() + STRIP_TIMEOUT_SECONDS
    stripped: Final = tuple((fragment, _strip(fragment, policy.strip_patterns, deadline)) for fragment in within_budget)
    unstripped: Final = len(distinct) - sum(1 for _, text in stripped if text is not None)
    if unstripped:
        verbose_proxy_logger.warning(
            "Generic Guardrail API (%s): %d text(s) are sent unstripped, because strip_patterns only run on the "
            "first %d characters of distinct text per guardrail call and for at most %s seconds.",
            guardrail_name,
            unstripped,
            MAX_STRIP_CALL_CHARS,
            STRIP_TIMEOUT_SECONDS,
        )
    return MappingProxyType({fragment: text for fragment, text in stripped if text is not None})


def _text_shaper(stripped: Mapping[str, str], max_text_chars: int | None) -> Callable[[str], str]:
    def shape(text: str) -> str:
        kept: Final = stripped.get(text, text)
        return kept if max_text_chars is None else kept[:max_text_chars]

    return shape


def _row_with_shaped_text(row: AllMessageValues, shape: Callable[[str], str]) -> AllMessageValues:
    return message_with_slot_texts(row, tuple(shape(text) for text in message_slot_texts(row))) or row


def shape_payload(
    dumped: Mapping[str, JsonValue], policy: PayloadPolicy, *, guardrail_name: str | None
) -> ShapedPayload:
    omitted: Final = policy.omitted_fields
    dumped_messages: Final = dumped.get("structured_messages")
    retained_messages: Final = _windowed_messages(dumped_messages, policy.max_messages)
    rows_windowed: Final = retained_messages is not dumped_messages
    retained_rows: Final = structured_messages_from_json(retained_messages) or ()
    dumped_texts: Final = dumped.get("texts")
    texts: Final = _string_list(dumped_texts)
    windowed_texts: Final = (
        tuple(chain.from_iterable(message_slot_texts(row) for row in retained_rows)) if rows_windowed else texts
    )
    shape: Final = _text_shaper(
        _stripped_fragments(
            chain(windowed_texts, chain.from_iterable(message_slot_texts(row) for row in retained_rows)),
            policy,
            guardrail_name,
        ),
        policy.max_text_chars,
    )
    sent_texts: Final = tuple(shape(text) for text in windowed_texts) if policy.shapes_text else windowed_texts
    texted_messages: Final = (
        _JSON.validate_python([_row_with_shaped_text(row, shape) for row in retained_rows])
        if policy.shapes_text and retained_rows
        else retained_messages
    )
    sent_messages: Final = (
        map_messages_image_urls(texted_messages, _omit_image) if policy.shapes_messages else texted_messages
    )
    shaped: Final = MappingProxyType(
        {
            **dumped,
            "structured_messages": sent_messages,
            "texts": None if dumped_texts is None else list(sent_texts),
        }
    )
    return ShapedPayload(
        body={key: value for key, value in shaped.items() if key not in omitted},
        sent_messages=structured_messages_from_json(sent_messages),
        loss=PayloadLoss(
            altered_message_indices=(
                frozenset() if rows_windowed else _altered_message_indices(dumped_messages, sent_messages)
            ),
            texts_omitted="texts" in omitted,
            messages_omitted="structured_messages" in omitted,
            images_omitted="images" in omitted,
            tools_omitted="tools" in omitted,
            text_shaped=rows_windowed or sent_texts != texts or texted_messages != retained_messages,
        ),
    )


def _null_free_object(pairs: Sequence[tuple[str, object]]) -> Mapping[str, object]:
    return MappingProxyType({key: item for key, item in pairs if item is not None})


def _without_nulls(value: object) -> object:
    normalized: Final[object] = json.loads(  # pyright: ignore[reportAny]  # stdlib parse of our own json.dumps output
        json.dumps(value), object_pairs_hook=_null_free_object
    )
    return normalized


def _rewrites(response: GenericGuardrailAPIResponse, body: Mapping[str, JsonValue]) -> bool:
    returned: Final = (
        ("texts", response.texts),
        ("structured_messages", response.structured_messages),
        ("images", response.images),
        ("tools", response.tools),
    )
    return any(value and _without_nulls(value) != _without_nulls(body.get(field)) for field, value in returned)


def block_only_response(
    response: GenericGuardrailAPIResponse,
    payload: ShapedPayload,
    *,
    input_type: Literal["request", "response"],
    guardrail_name: str | None,
) -> GenericGuardrailAPIResponse:
    if not payload.loss.text_shaped:
        return response
    if not _rewrites(response, payload.body):
        return GenericGuardrailAPIResponse(action=response.action, stream_holdback_chars=response.stream_holdback_chars)
    verbose_proxy_logger.warning(
        "Generic Guardrail API (%s): the guardrail rewrote a %s that max_messages, max_text_chars or strip_patterns "
        "shaped before it was sent. A rewrite of content it saw only in part cannot be applied, so the %s is "
        "rejected. These options are for block-only guardrails.",
        guardrail_name,
        input_type,
        input_type,
    )
    raise unappliable_request_rewrite(guardrail_name, input_type=input_type)


def _log_refused(field: str, detail: str, guardrail_name: str | None) -> None:
    verbose_proxy_logger.warning(
        "Generic Guardrail API (%s): ignoring the returned %s, %s.", guardrail_name, field, detail
    )


def _refused(returned: object, *, field: str, omitted: bool, guardrail_name: str | None) -> bool:
    if not returned or not omitted:
        return False
    _log_refused(field, "it was not sent to the guardrail", guardrail_name)
    return True


def accepted_rewrites(
    response: GenericGuardrailAPIResponse, loss: PayloadLoss, *, guardrail_name: str | None
) -> AcceptedRewrites:
    refused: Final = MappingProxyType(
        {
            field: _refused(returned, field=field, omitted=omitted, guardrail_name=guardrail_name)
            for field, returned, omitted in (
                ("texts", response.texts, loss.texts_omitted),
                ("images", response.images, loss.images_omitted),
                ("tools", response.tools, loss.tools_omitted),
                ("structured_messages", response.structured_messages, loss.messages_omitted),
            )
        }
    )
    return AcceptedRewrites(
        texts=None if refused["texts"] else response.texts or None,
        images=None if refused["images"] else response.images or None,
        tools=None if refused["tools"] else response.tools or None,
        rows=None if refused["structured_messages"] else response.structured_messages or None,
        refused_any=any(refused.values()),
    )


def _changes(accepted: object, posted: JsonValue) -> bool:
    return accepted is not None and accepted != posted


def raise_if_intervention_was_refused(
    *,
    action: str,
    accepted: AcceptedRewrites,
    posted: Mapping[str, JsonValue],
    rows_written_back: bool,
    input_type: Literal["request", "response"],
    guardrail_name: str | None,
) -> None:
    """A guardrail whose only real rewrite went to a field it was not sent would otherwise let the request
    through unchanged, so it is rejected instead. An echo of what the guardrail was sent changes nothing, and
    neither do returned images, since no endpoint writes them back. Tools and rows count only when the endpoint
    posted them, so a tool or row added to something the guardrail was never shown cannot stand in for the refused
    rewrite."""
    if action != "GUARDRAIL_INTERVENED" or not accepted.refused_any:
        return
    posted_tools: Final = posted.get("tools")
    applied: Final = (
        _changes(accepted.texts, posted.get("texts"))
        or (rows_written_back and posted.get("structured_messages") is not None)
        or (posted_tools is not None and _changes(accepted.tools, posted_tools))
    )
    if not applied:
        raise unappliable_request_rewrite(guardrail_name, input_type=input_type)


def _omitted_image_count(content: object) -> int:
    parts: Final = _part_list(content)
    return 0 if parts is None else sum(1 for part in parts if _holds_placeholder(part))


def _restored_part(returned: object, caller: object, sent: object) -> object:
    if returned == sent:
        return caller
    return (
        GENERIC_GUARDRAIL_UNAPPLIABLE_REWRITE if _holds_placeholder(sent) or _holds_placeholder(returned) else returned
    )


def _restored_content(returned: object, caller: object, sent: object) -> object:
    if returned == sent:
        return caller
    returned_parts: Final = _part_list(returned)
    caller_parts: Final = _part_list(caller)
    sent_parts: Final = _part_list(sent)
    if returned_parts is None or caller_parts is None or sent_parts is None:
        return GENERIC_GUARDRAIL_UNAPPLIABLE_REWRITE
    if not len(returned_parts) == len(caller_parts) == len(sent_parts):
        return GENERIC_GUARDRAIL_UNAPPLIABLE_REWRITE
    parts: Final = tuple(_restored_part(*aligned) for aligned in zip(returned_parts, caller_parts, sent_parts))
    restored: Final = [part for part in parts if not isinstance(part, GenericGuardrailUnappliableRewrite)]
    return restored if len(restored) == len(parts) else GENERIC_GUARDRAIL_UNAPPLIABLE_REWRITE


def _restored_row(
    returned: AllMessageValues,
    caller: AllMessageValues,
    sent: AllMessageValues,
    *,
    altered: bool,
) -> Mapping[str, object] | GenericGuardrailUnappliableRewrite:
    if returned is caller:
        return caller
    returned_content: Final = _content(returned)
    caller_content: Final = _content(caller)
    if not altered:
        adds_placeholder: Final = _omitted_image_count(returned_content) > _omitted_image_count(caller_content)
        return GENERIC_GUARDRAIL_UNAPPLIABLE_REWRITE if adds_placeholder else returned
    content: Final = _restored_content(returned_content, caller_content, _content(sent))
    if isinstance(content, GenericGuardrailUnappliableRewrite):
        return GENERIC_GUARDRAIL_UNAPPLIABLE_REWRITE
    return {**returned, "content": content}


def restore_unseen_rows(
    *,
    rows: tuple[AllMessageValues, ...] | None,
    caller: Sequence[AllMessageValues] | None,
    sent: Sequence[AllMessageValues] | None,
    loss: PayloadLoss,
    guardrail_name: str | None,
) -> tuple[AllMessageValues, ...] | None:
    """At a row the guardrail saw only in part, accept its rewrite only where the parts line up, and put the
    caller's image back into each part that still holds the placeholder. Anything else blocks the request,
    since keeping the caller's row would silently drop the guardrail's rewrite."""
    altered: Final = loss.altered_message_indices
    if rows is None or not altered:
        return rows
    if caller is None or sent is None or not len(rows) == len(caller) == len(sent):
        raise unappliable_request_rewrite(guardrail_name)
    restored: Final = tuple(
        _restored_row(row, caller_row, sent_row, altered=index in altered)
        for index, (row, caller_row, sent_row) in enumerate(zip(rows, caller, sent))
    )
    accepted: Final = structured_messages_from_json(
        [row for row in restored if not isinstance(row, GenericGuardrailUnappliableRewrite)]
    )
    if accepted is None or len(accepted) != len(restored):
        raise unappliable_request_rewrite(guardrail_name)
    return tuple(accepted)
