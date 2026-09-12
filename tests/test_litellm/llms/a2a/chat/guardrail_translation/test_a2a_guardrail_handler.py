"""Tests for litellm/llms/a2a/chat/guardrail_translation/handler.py."""

import json

from litellm.llms.a2a.chat.guardrail_translation.handler import A2AGuardrailHandler
from litellm.llms.base_llm.guardrail_translation.base_translation import StreamingScanKey


def _text_event(text: str) -> str:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "req-1",
            "result": {"kind": "message", "role": "agent", "parts": [{"kind": "text", "text": text}]},
        }
    )


def _status_event() -> str:
    return json.dumps({"jsonrpc": "2.0", "id": "req-1", "result": {"kind": "status-update", "status": {}}})


class TestA2AGuardrailHandlerStreamingScanKey:
    def test_key_joins_the_text_of_every_message_event(self):
        key = A2AGuardrailHandler().get_streaming_scan_key([_text_event("hello "), _text_event("world")])
        assert key == StreamingScanKey(texts=("hello world",))

    def test_events_without_text_leave_the_key_unchanged(self):
        handler = A2AGuardrailHandler()
        events = [_text_event("hello")]
        assert handler.get_streaming_scan_key(events + [_status_event()]) == handler.get_streaming_scan_key(events)

    def test_unparseable_items_are_ignored(self):
        key = A2AGuardrailHandler().get_streaming_scan_key([_text_event("hi"), "not json", b"bytes"])
        assert key.texts == ("hi",)
