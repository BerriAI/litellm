import json
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.exceptions import ModifyResponseException

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.guardrails import GuardrailEventHooks

_DEFAULT_VIOLATION_MESSAGE: Final = "I'm sorry, that request was blocked by the content filter."
BLOCKED_TOOL_OUTPUT: Final = json.dumps({"error": "Tool output blocked by content policy"})
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])
_JSON_ARRAY: Final = TypeAdapter(tuple[object, ...])
_TEXTS: Final = TypeAdapter(tuple[str, ...])
_TYPED_TEXT_PART_TYPES: Final = frozenset({"input_text", "output_text", "text"})


@dataclass(frozen=True, slots=True)
class RealtimeGuardrailBlock:
    message: str
    ends_session_on_violation: bool
    end_session_after_n_fails: int | None

    def ends_session(self, violation_count: int) -> bool:
        return self.ends_session_on_violation or (
            self.end_session_after_n_fails is not None and violation_count >= self.end_session_after_n_fails
        )


@dataclass(frozen=True, slots=True)
class RealtimeGuardrailRewrite:
    texts: tuple[str, ...]


RealtimeGuardrailVerdict: TypeAlias = RealtimeGuardrailBlock | RealtimeGuardrailRewrite | None


@dataclass(frozen=True, slots=True)
class RealtimeMessageText:
    role: str
    texts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RealtimeToolOutput:
    texts: tuple[str, ...]
    sanitized_event: str


RealtimeGuardedItem: TypeAlias = RealtimeMessageText | RealtimeToolOutput


def _json_object(value: object) -> Mapping[str, object] | None:
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


def _turn_copy(value: object) -> object:
    mapping: Final = _json_object(value)
    return value if mapping is None else dict(mapping)


def _typed_text(part: object) -> str | None:
    part_object: Final = _json_object(part)
    if part_object is None or part_object.get("type") not in _TYPED_TEXT_PART_TYPES:
        return None
    text: Final[object] = part_object.get("text", "")
    return text if isinstance(text, str) else None


def _content_parts(content: object) -> Sequence[object]:
    return _JSON_ARRAY.validate_python(content) if isinstance(content, list) else ()


def _typed_texts(content: object) -> tuple[str, ...]:
    return tuple(text for text in map(_typed_text, _content_parts(content)) if text is not None)


def _with_text(part: object, text: str | None) -> object:
    part_object: Final = _json_object(part)
    return part if text is None or part_object is None else {**part_object, "text": text}


def _tool_output(client_event: Mapping[str, object], item: Mapping[str, object]) -> RealtimeToolOutput | None:
    output: Final[object] = item.get("output", "")
    text: Final = output if isinstance(output, str) else json.dumps(output)
    if not text:
        return None
    sanitized_event: Final = json.dumps({**client_event, "item": {**item, "output": BLOCKED_TOOL_OUTPUT}})
    return RealtimeToolOutput(texts=(text,), sanitized_event=sanitized_event)


def realtime_guarded_item(client_event: Mapping[str, object]) -> RealtimeGuardedItem | None:
    if client_event.get("type") != "conversation.item.create":
        return None
    item: Final = _json_object(client_event.get("item"))
    if item is None:
        return None
    if item.get("type") == "function_call_output":
        return _tool_output(client_event, item)
    texts: Final = _typed_texts(item.get("content"))
    role: Final[object] = item.get("role")
    return RealtimeMessageText(role=role if isinstance(role, str) else "user", texts=texts) if any(texts) else None


def rewritten_item(client_event: Mapping[str, object], texts: tuple[str, ...]) -> Mapping[str, object]:
    item: Final = _json_object(client_event.get("item")) or {}
    if item.get("type") == "function_call_output":
        return {**item, "output": " ".join(texts)}
    parts: Final = _content_parts(item.get("content"))
    text_positions: Final = tuple(index for index, part in enumerate(parts) if _typed_text(part) is not None)
    replacements: Final = dict(zip(text_positions, texts))
    merged_away: Final = frozenset(text_positions[len(texts) :])
    return {
        **item,
        "content": [
            _with_text(part, replacements.get(index)) for index, part in enumerate(parts) if index not in merged_away
        ],
    }


def _violation_text(error: Exception) -> str:
    detail: Final[object] = getattr(error, "detail", None)
    detail_object: Final = _json_object(detail)
    if detail_object is not None:
        reason: Final[object] = detail_object.get("error")
        return reason if isinstance(reason, str) and reason else str(error)
    if detail is not None:
        return str(detail)
    return str(error) or _DEFAULT_VIOLATION_MESSAGE


def _is_guardrail_block(error: Exception) -> bool:
    return hasattr(error, "status_code") or isinstance(error, (ValueError, ModifyResponseException))


def _screened_texts(screened: object, texts: tuple[str, ...]) -> tuple[str, ...]:
    screened_object: Final = _json_object(screened)
    try:
        returned: Final = _TEXTS.validate_python(screened_object.get("texts") if screened_object else None)
    except ValidationError:
        return texts
    return returned if len(returned) == len(texts) else texts


async def _guardrail_outcome(
    guardrail: "CustomGuardrail", texts: tuple[str, ...], request_data: Mapping[str, object]
) -> RealtimeGuardrailBlock | tuple[str, ...]:
    try:
        screened: Final = await guardrail.apply_guardrail(
            inputs={"texts": list(texts), "images": []},
            request_data=dict(request_data),
            input_type="request",
        )
    except Exception as error:
        if not _is_guardrail_block(error):
            verbose_logger.exception("[realtime guardrail] unexpected error in apply_guardrail: %s", error)
            raise
        return RealtimeGuardrailBlock(
            message=guardrail.realtime_violation_message or _violation_text(error),
            ends_session_on_violation=guardrail.on_violation == "end_session",
            end_session_after_n_fails=guardrail.end_session_after_n_fails,
        )
    return _screened_texts(screened, texts)


async def _screen(
    guardrails: Sequence["CustomGuardrail"], texts: tuple[str, ...], request_data: Mapping[str, object]
) -> RealtimeGuardrailBlock | tuple[str, ...]:
    if not guardrails:
        return texts
    outcome: Final = await _guardrail_outcome(guardrails[0], texts, request_data)
    if isinstance(outcome, RealtimeGuardrailBlock):
        return outcome
    return await _screen(guardrails[1:], outcome, request_data)


async def _screened_until_block(
    guardrails: Sequence["CustomGuardrail"], texts: tuple[str, ...], request_data: Mapping[str, object]
) -> AsyncIterator[RealtimeGuardrailBlock | str]:
    for text in texts:
        if isinstance(outcome := await _screen(guardrails, (text,), request_data), RealtimeGuardrailBlock):
            yield outcome
            return
        for screened in outcome:
            yield screened


async def _screen_each(
    guardrails: Sequence["CustomGuardrail"], texts: tuple[str, ...], request_data: Mapping[str, object]
) -> RealtimeGuardrailBlock | tuple[str, ...]:
    outcomes: Final = tuple([outcome async for outcome in _screened_until_block(guardrails, texts, request_data)])
    block: Final = next((outcome for outcome in outcomes if isinstance(outcome, RealtimeGuardrailBlock)), None)
    return block if block is not None else tuple(outcome for outcome in outcomes if isinstance(outcome, str))


def _joins(texts: tuple[str, ...]) -> tuple[str, ...]:
    unspaced: Final = "".join(texts)
    boundaries: Final = zip(texts, texts[1:])
    word_break_unknown: Final = any(not left[-1:].isspace() and not right[:1].isspace() for left, right in boundaries)
    return (unspaced, " ".join(texts)) if word_break_unknown else (unspaced,)


async def _rewritten_parts(
    guardrails: Sequence["CustomGuardrail"], texts: tuple[str, ...], request_data: Mapping[str, object]
) -> RealtimeGuardrailBlock | tuple[str, ...]:
    parts: Final = await _screen_each(guardrails, texts, request_data)
    if isinstance(parts, RealtimeGuardrailBlock):
        return parts
    rejoined: Final = _joins(parts)
    screened_rejoined: Final = await _screen_each(guardrails, rejoined, request_data)
    if isinstance(screened_rejoined, RealtimeGuardrailBlock):
        return screened_rejoined
    masked_join: Final = next((text for text, join in zip(screened_rejoined, rejoined) if text != join), None)
    return parts if masked_join is None else (masked_join,)


def _in_role_scope(guardrail: "CustomGuardrail", role: str) -> bool:
    from litellm.llms.base_llm.guardrail_translation.utils import (
        effective_scan_only_tool_results_for_guardrail,
        effective_skip_system_message_for_guardrail,
        effective_skip_tool_message_for_guardrail,
        role_out_of_guardrail_scope,
    )

    return not role_out_of_guardrail_scope(
        role,
        skip_system_message=effective_skip_system_message_for_guardrail(guardrail),
        skip_tool_message=effective_skip_tool_message_for_guardrail(guardrail),
        scan_only_tool_results=effective_scan_only_tool_results_for_guardrail(guardrail),
    )


def realtime_runner_guardrail(
    callback: object,
    check_data: Mapping[str, object],
    event_hooks: Sequence["GuardrailEventHooks"],
) -> "CustomGuardrail | None":
    from litellm.integrations.custom_guardrail import CustomGuardrail

    if not isinstance(callback, CustomGuardrail) or callback.use_native_lifecycle_hooks:
        return None
    runs: Final = any(
        callback.should_run_guardrail(data=check_data, event_type=event_hook) for event_hook in event_hooks
    )
    return callback if runs else None


def realtime_screening_guardrails(
    role: str,
    event_hooks: Sequence["GuardrailEventHooks"],
    check_data: Mapping[str, object],
    callbacks: Iterable[object],
) -> tuple["CustomGuardrail", ...]:
    runner_guardrails: Final = (realtime_runner_guardrail(callback, check_data, event_hooks) for callback in callbacks)
    guardrails: Final = {
        id(guardrail): guardrail
        for guardrail in runner_guardrails
        if guardrail is not None and _in_role_scope(guardrail, role)
    }
    return tuple(guardrails.values())


async def realtime_guardrail_verdict(
    texts: tuple[str, ...],
    role: str,
    event_hooks: Sequence["GuardrailEventHooks"],
    request_data: Mapping[str, object],
    user_api_key_dict: object | None,
    callbacks: Iterable[object],
) -> RealtimeGuardrailVerdict:
    check_data: Final = {**request_data, "transcript": " ".join(texts), "stream": True}
    in_scope: Final = realtime_screening_guardrails(role, event_hooks, check_data, callbacks)
    guardrail_request_data: Final = {
        **{key: _turn_copy(value) for key, value in request_data.items()},
        "user_api_key_dict": user_api_key_dict,
    }
    joined: Final = _joins(texts)
    screened: Final = await _screen_each(in_scope, joined, guardrail_request_data)
    if isinstance(screened, RealtimeGuardrailBlock):
        return screened
    if screened == joined:
        return None
    rewritten: Final = screened if len(texts) < 2 else await _rewritten_parts(in_scope, texts, guardrail_request_data)
    if isinstance(rewritten, RealtimeGuardrailBlock):
        return rewritten
    return None if rewritten == texts else RealtimeGuardrailRewrite(texts=rewritten)


def guardrail_violation_error_event(message: str) -> str:
    return json.dumps(
        {
            "type": "error",
            "error": {"type": "guardrail_violation", "message": message, "code": "content_policy_violation"},
        }
    )


def guardrail_voice_prompt_events(message: str) -> tuple[str, str]:
    prompt: Final = f"Say exactly the following message to the user, word for word, do not add anything else: {message}"
    return (
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]},
            }
        ),
        json.dumps({"type": "response.create"}),
    )
