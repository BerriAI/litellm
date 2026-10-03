import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger

MODEL = "gpt-4o-mini"
PROMPT_TOKENS = 5
COMPLETION_TOKENS = 2


def _chat_completion_chunks() -> Tuple[Dict[str, Any], ...]:
    def chunk(delta: Dict[str, Any], finish_reason: Optional[str] = None) -> Dict[str, Any]:
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    return (
        chunk({"role": "assistant", "content": ""}),
        chunk({"content": "hello"}),
        chunk({}, finish_reason="stop"),
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
            "choices": [],
            "usage": {
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": COMPLETION_TOKENS,
                "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
            },
        },
    )


def _responses_api_chunks() -> Tuple[Dict[str, Any], ...]:
    in_progress = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": MODEL,
        "status": "in_progress",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }
    message_item = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "hello", "annotations": []}],
    }
    completed = {
        **in_progress,
        "status": "completed",
        "output": [message_item],
        "usage": {
            "input_tokens": PROMPT_TOKENS,
            "output_tokens": COMPLETION_TOKENS,
            "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
        },
    }
    return (
        {"type": "response.created", "response": in_progress, "sequence_number": 0},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 1,
            "item": {**message_item, "status": "in_progress", "content": []},
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "hello",
            "sequence_number": 2,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": message_item,
            "sequence_number": 3,
        },
        {"type": "response.completed", "response": completed, "sequence_number": 4},
    )


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        content_length = int(self.headers.get("content-length", 0))
        self.rfile.read(content_length)

        chunks = _responses_api_chunks() if self.path.endswith("/responses") else _chat_completion_chunks()

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        for chunk in chunks:
            self._write_chunk(f"data: {json.dumps(chunk)}\n\n".encode())
        self._write_chunk(b"data: [DONE]\n\n")
        self._write_chunk(b"")

    def _write_chunk(self, body: bytes) -> None:
        self.wfile.write(f"{len(body):X}\r\n".encode() + body + b"\r\n")
        self.wfile.flush()


class _RecordingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: List[Dict[str, Any]] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        standard_logging_object = kwargs.get("standard_logging_object")
        if standard_logging_object is not None:
            self.payloads.append(standard_logging_object)


async def _wait_for_payload(logger: _RecordingLogger) -> Dict[str, Any]:
    for _ in range(100):
        if logger.payloads:
            return logger.payloads[0]
        await asyncio.sleep(0.05)
    raise AssertionError("no success logging callback fired for the streamed /v1/messages request")


@pytest.mark.asyncio
@pytest.mark.parametrize("use_chat_completions_bridge", [True, False])
async def test_streaming_anthropic_messages_to_openai_backend_logs_usage(
    monkeypatch: pytest.MonkeyPatch, use_chat_completions_bridge: bool
) -> None:
    """
    Streaming /v1/messages against an ``openai/`` deployment must emit a success
    logging callback with real usage, on both bridges: the chat-completions
    adapter (``AnthropicStreamWrapper``) and the Responses API adapter
    (``AnthropicResponsesStreamWrapper``). Regression test for #35124, where the
    Responses bridge streamed a correct SSE body but never logged, so the
    request was billed by the provider and recorded with 0 tokens / $0 cost.
    """
    monkeypatch.setattr(
        litellm,
        "use_chat_completions_url_for_anthropic_messages",
        use_chat_completions_bridge,
    )
    logger = _RecordingLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])

    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockOpenAIHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        stream = await litellm.anthropic.messages.acreate(
            model=f"openai/{MODEL}",
            api_base=f"http://127.0.0.1:{server.server_port}",
            api_key="test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            stream=True,
        )
        events = [chunk async for chunk in stream]
        assert b"event: message_stop" in b"".join(events)

        payload = await _wait_for_payload(logger)
    finally:
        server.shutdown()
        server.server_close()

    assert payload["call_type"] == "anthropic_messages"
    assert payload["prompt_tokens"] == PROMPT_TOKENS
    assert payload["completion_tokens"] == COMPLETION_TOKENS
    assert payload["total_tokens"] == PROMPT_TOKENS + COMPLETION_TOKENS
    assert payload["response_cost"] > 0
