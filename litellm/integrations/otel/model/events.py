"""GenAI message events: the event half of ``capture_message_content``.

The OTel GenAI ``ContentCapturingMode`` records prompt/response content either as
span attributes (``gen_ai.input.messages`` / ``gen_ai.output.messages``, owned by
the mappers) or as events, or both. This module owns the event representation:
one ``gen_ai.{role}.message`` event per request message and one ``gen_ai.choice``
event per response choice, stamped on the LLM-call span so they ride whatever
trace exporter the operator already configured — the console exporter included —
without a separate logs pipeline.
"""

from typing import Mapping, cast

from litellm.integrations.otel.mappers.base import AttributeMap
from litellm.integrations.otel.mappers.utils import drop_none, message_content
from litellm.integrations.otel.model.payloads import LLMCallSpanData
from litellm.integrations.otel.model.semconv import GenAI, GenAIEvent
from litellm.integrations.otel.model.utils import as_str


def _role(message: object) -> str | None:
    """The chat-message ``role``, or ``None`` when absent or not a dict."""
    if not isinstance(message, dict):
        return None
    role = cast(Mapping[str, object], message).get("role")
    return role if isinstance(role, str) and role else None


def llm_message_events(
    data: LLMCallSpanData,
) -> tuple[tuple[str, AttributeMap], ...]:
    """``(event_name, attributes)`` pairs for an LLM call's prompt and response.

    One ``gen_ai.{role}.message`` event per request message and one
    ``gen_ai.choice`` per response choice, each carrying the provider, role, and
    (when present) the textual content. The caller stamps them as span events.
    """
    provider = data.provider or None
    inputs = tuple(_input_event(provider, m) for m in data.messages_in)
    outputs = tuple(
        _choice_event(provider, idx, c) for idx, c in enumerate(data.choices_out)
    )
    return inputs + outputs


def _input_event(
    provider: str | None, message: Mapping[str, object]
) -> tuple[str, AttributeMap]:
    role = _role(message) or "user"
    attrs = drop_none(
        {
            GenAI.PROVIDER_NAME: provider,
            "role": role,
            "content": message_content(message),
        }
    )
    return GenAIEvent.message(role), attrs


def _choice_event(
    provider: str | None, index: int, choice: Mapping[str, object]
) -> tuple[str, AttributeMap]:
    message = choice.get("message")
    attrs = drop_none(
        {
            GenAI.PROVIDER_NAME: provider,
            "index": index,
            "finish_reason": as_str(choice.get("finish_reason")),
            "role": _role(message),
            "content": message_content(message),
        }
    )
    return GenAIEvent.CHOICE, attrs
