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
class RealtimeUserText:
    text: str


@dataclass(frozen=True, slots=True)
class RealtimeToolOutput:
    text: str
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


def _input_texts(content: object) -> tuple[str, ...]:
    parts: Final[Sequence[object]] = _JSON_ARRAY.validate_python(content) if isinstance(content, list) else ()
    return tuple(text for text in map(_input_text, parts) if text is not None)


def _tool_output(client_event: Mapping[str, object], item: Mapping[str, object]) -> RealtimeToolOutput | None:
    output: Final[object] = item.get("output", "")
    text: Final = output if isinstance(output, str) else json.dumps(output)
    if not text:
        return None
    sanitized_event: Final = json.dumps({**client_event, "item": {**item, "output": BLOCKED_TOOL_OUTPUT}})
    return RealtimeToolOutput(text=text, sanitized_event=sanitized_event)


def realtime_guarded_item(client_event: Mapping[str, object]) -> RealtimeGuardedItem | None:
    """The client-controlled text a pre_call guardrail checks before a conversation.item.create reaches the model."""
    if client_event.get("type") != "conversation.item.create":
        return None
    item: Final = _json_object(client_event.get("item"))
    if item is None:
        return None
    if item.get("type") == "function_call_output":
        return _tool_output(client_event, item)
    if item.get("role") != "user":
        return None
    text: Final = " ".join(_input_texts(item.get("content")))
    return RealtimeUserText(text=text) if text else None


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


async def _guardrail_block(
    guardrail: "CustomGuardrail", text: str, user_api_key_dict: object | None
) -> RealtimeGuardrailBlock | None:
    try:
        await guardrail.apply_guardrail(
            inputs={"texts": [text], "images": []},
            request_data={"user_api_key_dict": user_api_key_dict},
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
    return None


async def realtime_guardrail_block(
    text: str,
    event_hooks: Sequence["GuardrailEventHooks"],
    request_data: Mapping[str, object],
    user_api_key_dict: object | None,
    callbacks: Iterable[object],
) -> RealtimeGuardrailBlock | None:
    """The first block any guardrail registered for ``event_hooks`` raises on realtime ``text``, or None when clean."""
    from litellm.integrations.custom_guardrail import CustomGuardrail

    check_data: Final = {**request_data, "transcript": text, "stream": True}
    guardrails: Final = {
        id(callback): callback
        for callback in callbacks
        if isinstance(callback, CustomGuardrail) and not callback.use_native_lifecycle_hooks
    }
    for guardrail in guardrails.values():
        if not any(guardrail.should_run_guardrail(data=check_data, event_type=hook) for hook in event_hooks):
            continue
        block = await _guardrail_block(guardrail, text, user_api_key_dict)
        if block is not None:
            return block
    return None


def guardrail_violation_error_event(message: str) -> str:
    return json.dumps(
        {
            "type": "error",
            "error": {"type": "guardrail_violation", "message": message, "code": "content_policy_violation"},
        }
    )


def guardrail_voice_prompt_events(message: str) -> tuple[str, str]:
    """The client events that make the model say the violation ``message`` aloud as its next response."""
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
