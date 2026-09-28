"""What the Generic Guardrail API endpoint is sent, and what it may write back.

Shaping is lossy, so every shaped payload carries a ``PayloadLoss``. Caller content the
guardrail did not see in full is never replaced by the guardrail's response.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.utils import unappliable_request_rewrite
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


@dataclass(frozen=True, slots=True)
class PayloadPolicy:
    send_images: bool = True
    exclude_fields: frozenset[str] = frozenset()

    @property
    def omitted_fields(self) -> frozenset[str]:
        return self.exclude_fields if self.send_images else self.exclude_fields | frozenset(("images",))

    @property
    def shapes_messages(self) -> bool:
        return not self.send_images

    @property
    def is_lossy(self) -> bool:
        return bool(self.omitted_fields)


@dataclass(frozen=True, slots=True)
class PayloadLoss:
    altered_message_indices: frozenset[int] = frozenset()
    texts_omitted: bool = False
    messages_omitted: bool = False
    images_omitted: bool = False
    tools_omitted: bool = False


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
    guardrail_name: str | None,
) -> PayloadPolicy:
    policy: Final = PayloadPolicy(
        send_images=_send_images(send_images),
        exclude_fields=_resolve_exclude_fields(
            config_values(exclude_payload_fields, option_name="exclude_payload_fields"), guardrail_name=guardrail_name
        ),
    )
    if policy.is_lossy:
        verbose_proxy_logger.warning(
            "Generic Guardrail API (%s): %s are not sent to the guardrail, so it can only enforce on what it is "
            "sent, and it cannot rewrite what it did not see.",
            guardrail_name,
            sorted(policy.omitted_fields),
        )
    return policy


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


def shape_payload(dumped: Mapping[str, JsonValue], policy: PayloadPolicy) -> ShapedPayload:
    omitted: Final = policy.omitted_fields
    dumped_messages: Final = dumped.get("structured_messages")
    sent_messages: Final = (
        map_messages_image_urls(dumped_messages, _omit_image) if policy.shapes_messages else dumped_messages
    )
    shaped: Final = MappingProxyType({**dumped, "structured_messages": sent_messages})
    return ShapedPayload(
        body={key: value for key, value in shaped.items() if key not in omitted},  # mutable-ok: JSON POST body
        sent_messages=structured_messages_from_json(sent_messages),
        loss=PayloadLoss(
            altered_message_indices=_altered_message_indices(dumped_messages, sent_messages),
            texts_omitted="texts" in omitted,
            messages_omitted="structured_messages" in omitted,
            images_omitted="images" in omitted,
            tools_omitted="tools" in omitted,
        ),
    )


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
