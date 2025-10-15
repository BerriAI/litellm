"""Common payload primitives for guardrail hooks.

These helpers focus on normalising incoming request structures so each
guardrail-specific hook can work with a consistent representation of the
user input. The goal is to avoid duplicating the same branching logic
(`messages` vs. `prompt`, etc.) across multiple providers while keeping
provider-specific behaviour (masking, exceptions, request mutation)
decoupled.

The design also leaves room for future response-side helpers so
`async_post_call_success_hook`-style flows can share the same patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

from litellm.types.llms.openai import AllMessageValues

GuardrailPayloadKind = Literal["messages", "text"]


@dataclass
class GuardrailPayload:
    """Normalised representation of data passed to a guardrail.

    Attributes:
        kind: Source shape for the payload (chat messages vs. raw text).
        messages: Original chat-style messages (if available).
        text_values: One or more plain-text entries extracted from the
            incoming request when no chat history exists.
        data_pointer: Key in the original `data` dict that should be
            updated when guardrail masking occurs (used for non-message
            payloads such as `prompt`).
        metadata: Extra context for callers (e.g., call type, original
            format). This keeps the structure flexible for future
            extensions, such as response payload handling.
    """

    kind: GuardrailPayloadKind
    messages: Optional[List[AllMessageValues]] = None
    text_values: Optional[List[str]] = None
    data_pointer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    had_messages_key: bool = False

    @property
    def is_empty(self) -> bool:
        if self.kind == "messages":
            return not self.messages
        return not self.text_values

    def to_message_list(self) -> List[AllMessageValues]:
        """Return a chat-style message list suitable for guardrail APIs."""

        if self.kind == "messages" and self.messages is not None:
            return self.messages

        message_list: List[AllMessageValues] = []
        for text in self.text_values or []:
            # We surface text as a synthetic user message to reuse existing
            # message-handling logic in guardrail implementations.
            message_list.append({"role": "user", "content": text})  # type: ignore[arg-type]
        return message_list

    def apply_masked_messages(
        self, data: Dict[str, Any], masked_messages: List[AllMessageValues]
    ) -> None:
        """Update the original request payload with guardrail-masked content."""

        if self.kind == "messages":
            data["messages"] = masked_messages
            return

        if self.kind == "text" and self.data_pointer:
            masked_texts: List[str] = []
            for message in masked_messages:
                text_value = self._extract_text_from_message(message)
                if text_value is not None:
                    masked_texts.append(text_value)

            if not masked_texts:
                if not self.had_messages_key:
                    data.pop("messages", None)
                return

            text_format = self.metadata.get("text_value_format", "single")
            if text_format == "list":
                data[self.data_pointer] = masked_texts
            else:
                data[self.data_pointer] = masked_texts[0]

            if not self.had_messages_key:
                data.pop("messages", None)

    @staticmethod
    def _extract_text_from_message(message: AllMessageValues) -> Optional[str]:
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                collected_parts: List[str] = []
                for block in content:
                    if isinstance(block, dict):
                        text_value = block.get("text")
                        if isinstance(text_value, str):
                            collected_parts.append(text_value)
                if collected_parts:
                    return "".join(collected_parts)
        return None

    @classmethod
    def from_messages(
        cls,
        messages: Optional[Sequence[AllMessageValues]],
        *,
        call_type: Optional[str] = None,
    ) -> Optional["GuardrailPayload"]:
        if not messages:
            return None

        payload = cls(kind="messages", messages=list(messages))
        if call_type is not None:
            payload.metadata.setdefault("call_type", call_type)
        return payload

    @classmethod
    def from_text(
        cls,
        text_value: Optional[Union[Iterable[str], str]],
        *,
        data_pointer: str,
        call_type: Optional[str] = None,
    ) -> Optional["GuardrailPayload"]:
        if text_value is None:
            return None

        if isinstance(text_value, str):
            if not text_value:
                return None
            text_values = [text_value]
            text_format = "single"
        else:
            text_values = [
                text for text in text_value if isinstance(text, str) and text
            ]
            if not text_values:
                return None
            text_format = "list" if len(text_values) > 1 else "single"

        payload = cls(
            kind="text",
            text_values=text_values,
            data_pointer=data_pointer,
            metadata={"text_value_format": text_format},
        )
        if call_type is not None:
            payload.metadata.setdefault("call_type", call_type)
        return payload


GuardrailInputExtractor = Callable[[Dict[str, Any], str], Optional[GuardrailPayload]]


def _extract_image_prompt_payload(
    data: Dict[str, Any], call_type: str
) -> Optional[GuardrailPayload]:
    prompt = data.get("prompt")
    return GuardrailPayload.from_text(
        prompt, data_pointer="prompt", call_type=call_type
    )


PAYLOAD_EXTRACTORS: Dict[str, GuardrailInputExtractor] = {
    "image_generation": _extract_image_prompt_payload,
}


def get_guardrail_input_payload(
    data: Optional[Dict[str, Any]],
    call_type: str,
) -> Optional[GuardrailPayload]:
    """Return a normalised guardrail payload for the given request.

    The lookup order prefers chat messages (if available) and falls back to
    call-type specific extractors. This keeps inputs deterministic and makes
    it easy to add new modalities (audio, tool calls, etc.) by registering
    additional extractors.
    """

    if not isinstance(data, dict):
        return None

    has_messages_key = "messages" in data
    messages_value = data["messages"] if has_messages_key else None

    message_payload = GuardrailPayload.from_messages(
        messages_value, call_type=call_type
    )
    if message_payload is not None and not message_payload.is_empty:
        message_payload.had_messages_key = has_messages_key
        return message_payload

    extractor = PAYLOAD_EXTRACTORS.get(call_type)
    if extractor:
        payload = extractor(data, call_type)
        if payload is not None and not payload.is_empty:
            payload.had_messages_key = has_messages_key
            return payload

    return None


__all__ = [
    "GuardrailPayload",
    "GuardrailPayloadKind",
    "get_guardrail_input_payload",
]
