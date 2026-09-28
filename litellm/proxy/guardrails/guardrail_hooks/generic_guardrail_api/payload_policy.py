"""What the Generic Guardrail API endpoint is sent, and what it may write back.

Shaping is lossy, so every shaped payload carries a ``PayloadLoss``. Caller content the
guardrail did not see in full is never replaced by the guardrail's response.
"""

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import accumulate, chain
from types import MappingProxyType
from typing import Final, Literal, assert_never

import regex
from pydantic import JsonValue

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.llms.base_llm.guardrail_translation.utils import (
    message_slot_texts,
    message_with_slot_texts,
    unappliable_request_rewrite,
)
from litellm.proxy.guardrails._content_utils import as_json_value, image_part_url, map_messages_image_urls
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
    GenericGuardrailAPIRequest,
    GenericGuardrailAPIResponse,
    GuardrailToolParam,
    structured_messages_from_json,
)

from .config_parsing import config_values

PROTECTED_PAYLOAD_FIELDS: Final = frozenset({"input_type", "litellm_call_id"})

IMAGE_OMITTED_PLACEHOLDER: Final = "[omitted]"

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


@dataclass(frozen=True, slots=True)
class _Unappliable:
    pass


_UNAPPLIABLE: Final = _Unappliable()


def resolve_payload_policy(
    *,
    send_images: object,
    exclude_payload_fields: Sequence[str] | None,
    max_messages: object,
    max_text_chars: object,
    strip_patterns: Sequence[str] | None,
    guardrail_name: str | None,
) -> PayloadPolicy:
    policy: Final = PayloadPolicy(
        send_images=_send_images(send_images),
        exclude_fields=_resolve_exclude_fields(
            config_values(exclude_payload_fields, option_name="exclude_payload_fields"), guardrail_name=guardrail_name
        ),
        max_messages=_positive_int(max_messages, option_name="max_messages"),
        max_text_chars=_positive_int(max_text_chars, option_name="max_text_chars"),
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


def _positive_int(value: object, *, option_name: str) -> int | None:
    match value:
        case None:
            return None
        case bool():
            raise ValueError(f"{option_name} must be an int, got {value!r}")
        case int() if value >= 1:
            return value
        case int():
            raise ValueError(f"{option_name} must be >= 1 (got {value})")
        case _:
            raise ValueError(f"{option_name} must be an int, got {value!r}")


def _compile_strip_patterns(raw: Sequence[str] | None) -> tuple[regex.Pattern[str], ...]:
    try:
        return tuple(regex.compile(pattern) for pattern in config_values(raw, option_name="strip_patterns"))
    except regex.error as e:
        raise ValueError(f"strip_patterns contains an invalid regex: {e}") from e


def _send_images(value: object) -> bool:
    match value:
        case None:
            return True
        case bool():
            return value
        case _:
            raise ValueError(f"send_images must be a bool, got {value!r}")


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
    return IMAGE_OMITTED_PLACEHOLDER


def _content(row: JsonValue) -> JsonValue:
    return row.get("content") if isinstance(row, dict) else None


def _holds_placeholder(part: JsonValue) -> bool:
    return image_part_url(part) == IMAGE_OMITTED_PLACEHOLDER


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
        as_json_value([_row_with_shaped_text(row, shape) for row in retained_rows])
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
            "texts": None if dumped_texts is None else list(sent_texts),  # mutable-ok: JSON texts is an array
        }
    )
    return ShapedPayload(
        body={key: value for key, value in shaped.items() if key not in omitted},  # mutable-ok: JSON POST body
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


def _without_nulls(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: _without_nulls(item) for key, item in value.items() if item is not None}  # mutable-ok: JSON
    if isinstance(value, list):
        return [_without_nulls(item) for item in value]  # mutable-ok: JSON array
    return value


def _rewrites(response: GenericGuardrailAPIResponse, body: Mapping[str, JsonValue]) -> bool:
    returned: Final = (
        ("texts", response.texts),
        ("structured_messages", response.structured_messages),
        ("images", response.images),
        ("tools", response.tools),
    )
    return response.action == "GUARDRAIL_INTERVENED" or any(
        value and _without_nulls(as_json_value(value)) != _without_nulls(body.get(field)) for field, value in returned
    )


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
    match input_type:
        case "request":
            raise unappliable_request_rewrite(guardrail_name)
        case "response":
            raise GuardrailRaisedException(
                guardrail_name=guardrail_name,
                message=f"Guardrail '{guardrail_name}' returned a rewrite that cannot be applied to this response",
                should_wrap_with_default_message=False,
            )
        case _:
            assert_never(input_type)


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


def _changes(accepted: object, original: object) -> bool:
    return accepted is not None and as_json_value(accepted) != as_json_value(original)


def raise_if_intervention_was_refused(
    *,
    action: str,
    accepted: AcceptedRewrites,
    original_texts: Sequence[str],
    original_images: Sequence[str] | None,
    original_tools: Sequence[Mapping[str, object]] | None,
    rows_written_back: bool,
    guardrail_name: str | None,
) -> None:
    """A guardrail whose only real rewrite went to a field it was not sent would otherwise let the request
    through unchanged, so it is rejected instead. An echo of what the caller sent changes nothing."""
    applied: Final = rows_written_back or any(
        _changes(accepted_value, original_value)
        for accepted_value, original_value in (
            (accepted.texts, original_texts),
            (accepted.images, original_images),
            (accepted.tools, original_tools),
        )
    )
    if action == "GUARDRAIL_INTERVENED" and accepted.refused_any and not applied:
        raise unappliable_request_rewrite(guardrail_name)


def _omitted_image_count(content: JsonValue) -> int:
    if not isinstance(content, list):
        return 0
    return sum(1 for part in content if _holds_placeholder(part))


def _restored_part(returned: JsonValue, caller: JsonValue, sent: JsonValue) -> JsonValue | _Unappliable:
    if returned == sent:
        return caller
    return _UNAPPLIABLE if _holds_placeholder(sent) or _holds_placeholder(returned) else returned


def _restored_content(returned: JsonValue, caller: JsonValue, sent: JsonValue) -> JsonValue | _Unappliable:
    if returned == sent:
        return caller
    if not isinstance(returned, list) or not isinstance(caller, list) or not isinstance(sent, list):
        return _UNAPPLIABLE
    if not len(returned) == len(caller) == len(sent):
        return _UNAPPLIABLE
    parts: Final = tuple(_restored_part(*aligned) for aligned in zip(returned, caller, sent))
    restored: Final = [part for part in parts if not isinstance(part, _Unappliable)]  # mutable-ok: JSON array
    return restored if len(restored) == len(parts) else _UNAPPLIABLE


def _restored_row(
    returned: AllMessageValues,
    caller: AllMessageValues,
    sent: AllMessageValues,
    *,
    altered: bool,
) -> AllMessageValues | JsonValue | _Unappliable:
    if returned is caller:
        return caller
    returned_json: Final = as_json_value(returned)
    caller_content: Final = _content(as_json_value(caller))
    if not altered:
        adds_placeholder: Final = _omitted_image_count(_content(returned_json)) > _omitted_image_count(caller_content)
        return _UNAPPLIABLE if adds_placeholder else returned
    content: Final = _restored_content(_content(returned_json), caller_content, _content(as_json_value(sent)))
    if isinstance(content, _Unappliable) or not isinstance(returned_json, dict):
        return _UNAPPLIABLE
    return {**returned_json, "content": content}  # mutable-ok: JSON row


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
        [row for row in restored if not isinstance(row, _Unappliable)]  # mutable-ok: JSON array
    )
    if accepted is None or len(accepted) != len(restored):
        raise unappliable_request_rewrite(guardrail_name)
    return tuple(accepted)
