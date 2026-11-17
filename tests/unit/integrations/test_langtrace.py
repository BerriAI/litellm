import json
from collections.abc import Sequence
from typing import Final

import pytest
from opentelemetry.sdk.trace import TracerProvider

from litellm.integrations.langtrace import LangtraceAttributes
from litellm.types.integrations.otel_span_attributes import SpanAttributes
from litellm.types.utils import Message, ModelResponse

CHAT_MESSAGE_DICTS: Final = ({"role": "user", "content": "weather in SF?"}, {"role": "assistant", "content": "Sunny."})
CHAT_MESSAGE_OBJECTS: Final = tuple(
    Message(role=message["role"], content=message["content"]) for message in CHAT_MESSAGE_DICTS
)
NOT_JSON_ENCODABLE: Final = object()
NON_UTF8_BYTES: Final = b"\x80"


@pytest.mark.parametrize(
    ("messages", "expected_prompts"),
    [
        (CHAT_MESSAGE_DICTS, list(CHAT_MESSAGE_DICTS)),
        (CHAT_MESSAGE_OBJECTS, [message.model_dump() for message in CHAT_MESSAGE_OBJECTS]),
        (
            ({"role": "user", "content": "before\x00after", "metadata": {1: "one"}},),
            [{"role": "user", "content": "before\x00after", "metadata": {"1": "one"}}],
        ),
        (
            (Message(role="user", content="weather in SF?", trace_handle=NOT_JSON_ENCODABLE),),
            [{**CHAT_MESSAGE_OBJECTS[0].model_dump(), "trace_handle": str(NOT_JSON_ENCODABLE)}],
        ),
        (
            (Message(role="user", content="weather in SF?", audio_bytes=NON_UTF8_BYTES),),
            [str(Message(role="user", content="weather in SF?", audio_bytes=NON_UTF8_BYTES))],
        ),
    ],
    ids=[
        "dict_messages",
        "litellm_message_objects",
        "dict_message_with_nul_and_int_key",
        "litellm_message_with_value_json_cannot_encode",
        "litellm_message_with_non_utf8_bytes",
    ],
)
def test_langtrace_span_keeps_prompt_and_token_usage(
    messages: Sequence[object], expected_prompts: Sequence[object]
) -> None:
    span: Final = TracerProvider().get_tracer("langtrace-tests").start_span("litellm_request")
    response: Final = ModelResponse(
        model="gpt-4o",
        choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Still sunny."}}],
        usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    )

    LangtraceAttributes().set_langtrace_attributes(
        span, {"messages": messages, "litellm_params": {"custom_llm_provider": "openai"}}, response
    )

    assert [
        json.loads(str((event.attributes or {})[SpanAttributes.LLM_PROMPTS.value]))
        for event in span.events
        if event.name == "gen_ai.content.prompt"
    ] == [expected_prompts]
    assert {
        key: (span.attributes or {}).get(key)
        for key in (
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS.value,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS.value,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS.value,
        )
    } == {
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS.value: 7,
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS.value: 3,
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS.value: 10,
    }
