import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.guardrails import GuardrailEventHooks

_DEFAULT_VIOLATION_MESSAGE: Final = "I'm sorry, that request was blocked by the content filter."
BLOCKED_TOOL_OUTPUT: Final = json.dumps({"error": "Tool output blocked by content policy"})
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])
_JSON_ARRAY: Final = TypeAdapter(tuple[object, ...])
_TEXTS: Final = TypeAdapter(tuple[str, ...])


@dataclass(frozen=True, slots=True)
class RealtimeGuardrailBlock:
    message: str
    spoken_message: str
    ends_session_on_violation: bool
    end_session_after_n_fails: int | None

    def ends_session(self, violation_count: int) -> bool:
        return self.ends_session_on_violation or (
            self.end_session_after_n_fails is not None and violation_count >= self.end_session_after_n_fails
        )


GUARDRAIL_FAILURE_BLOCK: Final = RealtimeGuardrailBlock(
    message=_DEFAULT_VIOLATION_MESSAGE,
    spoken_message=_DEFAULT_VIOLATION_MESSAGE,
    ends_session_on_violation=True,
    end_session_after_n_fails=None,
)


@dataclass(frozen=True, slots=True)
class RealtimeGuardrailRewrite:
    texts: tuple[str, ...]


RealtimeGuardrailVerdict: TypeAlias = RealtimeGuardrailBlock | RealtimeGuardrailRewrite | None


@dataclass(frozen=True, slots=True)
class RealtimeUserText:
    texts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RealtimeToolOutput:
    texts: tuple[str, ...]
    sanitized_event: str


RealtimeGuardedItem: TypeAlias = RealtimeUserText | RealtimeToolOutput


def _json_object(value: object) -> Mapping[str, object] | None:
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


def _input_text(part: object) -> str | None:
    part_object: Final = _json_object(part)
    if part_object is None or part_object.get("type") != "input_text":
        return None
    text: Final[object] = part_object.get("text", "")
    return text if isinstance(text, str) else None


def _content_parts(content: object) -> Sequence[object]:
    return _JSON_ARRAY.validate_python(content) if isinstance(content, list) else ()


def _input_texts(content: object) -> tuple[str, ...]:
    return tuple(text for text in map(_input_text, _content_parts(content)) if text is not None)


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
    texts: Final = _input_texts(item.get("content"))
    return RealtimeUserText(texts=texts) if any(texts) else None


def rewritten_item(client_event: Mapping[str, object], texts: tuple[str, ...]) -> Mapping[str, object]:
    item: Final = _json_object(client_event.get("item")) or {}
    if item.get("type") == "function_call_output":
        return {**item, "output": " ".join(texts)}
    parts: Final = _content_parts(item.get("content"))
    text_positions: Final = tuple(index for index, part in enumerate(parts) if _input_text(part) is not None)
    replacements: Final = dict(zip(text_positions, texts))
    return {**item, "content": [_with_text(part, replacements.get(index)) for index, part in enumerate(parts)]}


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
    return hasattr(error, "status_code") or isinstance(error, ValueError)


def _screened_texts(screened: object, texts: tuple[str, ...]) -> tuple[str, ...]:
    screened_object: Final = _json_object(screened)
    try:
        returned: Final = _TEXTS.validate_python(screened_object.get("texts") if screened_object else None)
    except ValidationError:
        return texts
    return returned if len(returned) == len(texts) else texts


async def _guardrail_outcome(
    guardrail: "CustomGuardrail", texts: tuple[str, ...], user_api_key_dict: object | None
) -> RealtimeGuardrailBlock | tuple[str, ...]:
    try:
        screened: Final = await guardrail.apply_guardrail(
            inputs={"texts": list(texts), "images": []},
            request_data={"user_api_key_dict": user_api_key_dict},
            input_type="request",
        )
    except Exception as error:
        if not _is_guardrail_block(error):
            verbose_logger.exception("[realtime guardrail] unexpected error in apply_guardrail: %s", error)
            raise
        return RealtimeGuardrailBlock(
            message=guardrail.realtime_violation_message or _violation_text(error),
            spoken_message=guardrail.realtime_violation_message or _DEFAULT_VIOLATION_MESSAGE,
            ends_session_on_violation=guardrail.on_violation == "end_session",
            end_session_after_n_fails=guardrail.end_session_after_n_fails,
        )
    return _screened_texts(screened, texts)


async def _screen(
    guardrails: Sequence["CustomGuardrail"],
    texts: tuple[str, ...],
    original_texts: tuple[str, ...],
    user_api_key_dict: object | None,
) -> RealtimeGuardrailVerdict:
    if not guardrails:
        return None if texts == original_texts else RealtimeGuardrailRewrite(texts=texts)
    outcome: Final = await _guardrail_outcome(guardrails[0], texts, user_api_key_dict)
    if isinstance(outcome, RealtimeGuardrailBlock):
        return outcome
    return await _screen(guardrails[1:], outcome, original_texts, user_api_key_dict)


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


async def realtime_guardrail_verdict(
    texts: tuple[str, ...],
    event_hooks: Sequence["GuardrailEventHooks"],
    request_data: Mapping[str, object],
    user_api_key_dict: object | None,
    callbacks: Iterable[object],
) -> RealtimeGuardrailVerdict:
    check_data: Final = {**request_data, "transcript": " ".join(texts), "stream": True}
    runner_guardrails: Final = (realtime_runner_guardrail(callback, check_data, event_hooks) for callback in callbacks)
    guardrails: Final = {id(guardrail): guardrail for guardrail in runner_guardrails if guardrail is not None}
    return await _screen(tuple(guardrails.values()), texts, texts, user_api_key_dict)


def guardrail_violation_error_event(message: str) -> str:
    return json.dumps(
        {
            "type": "error",
            "error": {"type": "guardrail_violation", "message": message, "code": "content_policy_violation"},
        }
    )


def guardrail_voice_prompt_events(message: str) -> tuple[str, str]:
    prompt: Final = (
        f'Say exactly the following message to the user, word for word, do not add anything else: "{message}"'
    )
    return (
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]},
            }
        ),
        json.dumps({"type": "response.create"}),
    )
