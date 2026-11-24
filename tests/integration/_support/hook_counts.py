"""Scripted provider, proxy config and request helpers shared by the CustomLogger hook count tests."""

import base64
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import httpx
import yaml
from integration._support.client import Gateway, held
from integration._support.wire import Reply, Request, Wire

WAV_BYTES: Final = (
    b"RIFF\x30\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x40\x1f\x00\x00\x40\x1f\x00\x00\x01\x00\x08\x00data\x08\x00\x00\x00"
    b"\x00\x01\x02\x03\x04\x05\x06\x07"
)
PNG_BYTES: Final = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\rIDAT\x08\xd7c\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff"
    b"\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)
_IMAGE_REPLY: Final = json.dumps({"created": 1700000000, "data": [{"b64_json": base64.b64encode(PNG_BYTES).decode()}]})
_CHAT_REPLY: Final = json.dumps(
    {
        "id": "chatcmpl-hook-count",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
)
_EMBEDDING_REPLY: Final = json.dumps(
    {
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5]}],
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }
)
_MESSAGE_REPLY: Final = json.dumps(
    {
        "id": "msg_hook_count",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }
)
_RESPONSE_REPLY: Final = json.dumps(
    {
        "id": "resp_hook_count",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "id": "msg_hook_count",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
    }
)
_OCR_REPLY: Final = json.dumps(
    {
        "pages": [
            {"index": 0, "markdown": "page", "images": [], "dimensions": {"dpi": 200, "height": 1100, "width": 850}}
        ],
        "model": "mistral-ocr-latest",
        "usage_info": {"pages_processed": 1, "doc_size_bytes": 64},
        "document_annotation": None,
    }
)
_SEARCH_REPLY: Final = json.dumps(
    {"results": [{"title": "hook count", "url": "https://result.invalid/a", "snippet": "hit"}]}
)
_CHAT_FRAMES: Final = (
    'data: {"id":"chatcmpl-hook-count","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}',
    'data: {"id":"chatcmpl-hook-count","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    '"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}',
    "data: [DONE]",
)
_MESSAGE_FRAMES: Final = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"id":"msg_hook_count","type":"message","role":"assistant",'
    '"model":"claude-haiku-4-5","content":[],"stop_reason":null,"stop_sequence":null,'
    '"usage":{"input_tokens":3,"output_tokens":0}}}',
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}',
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
    "event: message_delta\n"
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}',
    'event: message_stop\ndata: {"type":"message_stop"}',
)
_RESPONSE_FRAMES: Final = (
    "event: response.created\n"
    'data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_hook_count","object":"response",'
    '"created_at":1,"status":"in_progress","model":"gpt-4o-mini","output":[],"usage":null}}',
    "event: response.output_text.delta\n"
    'data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_hook_count","output_index":0,'
    '"content_index":0,"delta":"ok"}',
    "event: response.completed\n"
    'data: {"type":"response.completed","sequence_number":2,"response":{"id":"resp_hook_count","object":"response",'
    '"created_at":1,"status":"completed","model":"gpt-4o-mini","output":[{"type":"message","id":"msg_hook_count",'
    '"status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}],'
    '"usage":{"input_tokens":3,"output_tokens":1,"total_tokens":4}}}',
)
_STREAM_FRAMES: Final = {
    "/v1/chat/completions": _CHAT_FRAMES,
    "/v1/messages": _MESSAGE_FRAMES,
    "/v1/responses": _RESPONSE_FRAMES,
}
SURFACE_CALL_TYPES: Final = {
    "chat": "acompletion",
    "embedding": "aembedding",
    "messages": "anthropic_messages",
    "responses": "aresponses",
    "speech": "aspeech",
    "transcription": "atranscription",
    "image_generation": "aimage_generation",
    "image_edit": "aimage_edit",
    "ocr": "aocr",
    "search": "asearch",
}
IN_THE_ASYNC_FLAG_LIST: Final = frozenset({"chat", "embedding", "messages", "responses"})
STREAMING_SURFACES: Final = ("chat", "messages", "responses")


def _streams(request: Request) -> bool:
    if request.headers.get("content-type", "").startswith("application/json"):
        return json.loads(request.body or b"{}").get("stream") is True
    return False


def provider(request: Request) -> Reply:
    """Answers every surface the hook count tests drive; any target under /fail/ is a provider 400."""
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    if request.target.startswith("/fail/"):
        return Reply(
            status=400,
            body=b'{"error":{"message":"scripted provider rejection","type":"invalid_request_error","code":null}}',
        )
    if request.target in _STREAM_FRAMES and _streams(request):
        return Reply(
            chunks=tuple(f"{frame}\n\n".encode() for frame in _STREAM_FRAMES[request.target]),
            content_type="text/event-stream",
        )
    replies: Final[dict[str, Callable[[], Reply]]] = {
        "/v1/chat/completions": lambda: Reply(body=_CHAT_REPLY.encode()),
        "/v1/embeddings": lambda: Reply(body=_EMBEDDING_REPLY.encode()),
        "/v1/messages": lambda: Reply(body=_MESSAGE_REPLY.encode()),
        "/v1/responses": lambda: Reply(body=_RESPONSE_REPLY.encode()),
        "/v1/audio/speech": lambda: Reply(body=WAV_BYTES, content_type="audio/wav"),
        "/v1/audio/transcriptions": lambda: Reply(body=b'{"text":"hook count transcript"}'),
        "/v1/images/generations": lambda: Reply(body=_IMAGE_REPLY.encode()),
        "/v1/images/edits": lambda: Reply(body=_IMAGE_REPLY.encode()),
        "/v1/ocr": lambda: Reply(body=_OCR_REPLY.encode()),
        "/perplexity/search": lambda: Reply(body=_SEARCH_REPLY.encode()),
    }
    return replies[request.target]()


@dataclass
class HookRecords:
    """Every hook record the probe reported, kept across drains so cells can read their own call id."""

    wire: Wire
    function_callbacks: bool
    seen: list[dict[str, str]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def drained(self) -> tuple[dict[str, str], ...]:
        with self.lock:
            self.seen.extend(json.loads(request.body) for request in self.wire.drain())
            return tuple(self.seen)

    def for_call(self, call_id: str) -> tuple[dict[str, str], ...]:
        return tuple(record for record in self.drained() if record["litellm_call_id"] == call_id)


def _model(name: str, model: str, api_base: str, **extra: object) -> dict[str, object]:
    return {
        "model_name": name,
        "litellm_params": {"model": model, "api_base": api_base, "api_key": "synthetic-provider-key"},
        **extra,
    }


def write_hook_count_config(
    path: Path,
    provider_url: str,
    litellm_settings: Mapping[str, object],
    guardrails: tuple[Mapping[str, object], ...] = (),
) -> Path:
    """One success and one failure deployment per surface, all answered by `provider`."""
    models: Final = []
    tools: Final = []
    for outcome, prefix in (("success", ""), ("failure", "/fail")):
        base: str = provider_url + prefix
        models.extend(
            (
                _model(f"hooks-chat-{outcome}", "openai/gpt-4o-mini", base + "/v1"),
                _model(f"hooks-embedding-{outcome}", "openai/text-embedding-3-small", base + "/v1"),
                _model(f"hooks-messages-{outcome}", "anthropic/claude-haiku-4-5", base),
                _model(f"hooks-responses-{outcome}", "openai/gpt-4o-mini", base + "/v1"),
                _model(f"hooks-speech-{outcome}", "openai/gpt-4o-mini-tts", base + "/v1"),
                _model(f"hooks-transcription-{outcome}", "openai/whisper-1", base + "/v1"),
                _model(f"hooks-image_generation-{outcome}", "openai/gpt-image-1", base + "/v1"),
                _model(f"hooks-image_edit-{outcome}", "openai/gpt-image-1", base + "/v1"),
                _model(f"hooks-ocr-{outcome}", "mistral/mistral-ocr-latest", base + "/v1", model_info={"mode": "ocr"}),
            )
        )
        tools.append(
            {
                "search_tool_name": f"hooks-search-{outcome}",
                "litellm_params": {
                    "search_provider": "perplexity",
                    "api_key": "synthetic-provider-key",
                    "api_base": base + "/perplexity",
                },
            }
        )
    path.write_text(
        yaml.safe_dump(
            {
                "model_list": models,
                "search_tools": tools,
                "litellm_settings": dict(litellm_settings),
                **({"guardrails": [dict(guardrail) for guardrail in guardrails]} if guardrails else {}),
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "disable_model_info_refresh": True,
                },
                "router_settings": {"num_retries": 0},
            }
        )
    )
    return path


def send(
    proxy: Gateway, surface: str, outcome: str, call_id: str, *, key: str | None = None, stream: bool = False
) -> httpx.Response:
    model: Final = f"hooks-{surface}-{outcome}"
    nonce: Final = f"hook-count-{call_id}"
    headers: Final = {"Authorization": f"Bearer {key or proxy.key}", "x-litellm-call-id": call_id}
    if surface == "transcription":
        return proxy.client.post(
            "/v1/audio/transcriptions",
            data={"model": model, "prompt": nonce},
            files={"file": ("a.wav", WAV_BYTES + call_id.encode(), "audio/wav")},
            headers=headers,
        )
    if surface == "image_edit":
        return proxy.client.post(
            "/v1/images/edits",
            data={"model": model, "prompt": nonce},
            files={"image": ("a.png", PNG_BYTES, "image/png")},
            headers=headers,
        )
    streaming: Final[dict[str, object]] = {"stream": True} if stream else {}
    routes: Final[dict[str, tuple[str, dict[str, object]]]] = {
        "chat": (
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": nonce}], **streaming},
        ),
        "embedding": ("/v1/embeddings", {"model": model, "input": nonce}),
        "messages": (
            "/v1/messages",
            {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": nonce}], **streaming},
        ),
        "responses": ("/v1/responses", {"model": model, "input": nonce, **streaming}),
        "speech": ("/v1/audio/speech", {"model": model, "input": nonce, "voice": "alloy", "response_format": "wav"}),
        "image_generation": ("/v1/images/generations", {"model": model, "prompt": nonce, "n": 1}),
        "ocr": (
            "/v1/ocr",
            {
                "model": model,
                "document": {
                    "type": "document_url",
                    "document_url": "data:application/pdf;base64," + base64.b64encode(nonce.encode()).decode(),
                },
            },
        ),
        "search": ("/v1/search", {"search_tool_name": f"hooks-search-{outcome}", "query": nonce}),
    }
    path, body = routes[surface]
    return proxy.client.post(path, json=body, headers=headers)


def counts_function_hooks(records: HookRecords, surface: str) -> bool:
    return records.function_callbacks and surface not in IN_THE_ASYNC_FLAG_LIST


def probe_hooks(records: HookRecords, surface: str, found: tuple[dict[str, str], ...]) -> list[str]:
    counted: Final = counts_function_hooks(records, surface)
    return sorted(record["hook"] for record in found if counted or not record["hook"].startswith("function_"))


def one_async_hook(records: HookRecords, surface: str, outcome: str) -> list[str]:
    function_hook: Final = (f"function_{outcome}",) if counts_function_hooks(records, surface) else ()
    return sorted((f"async_{outcome}", *function_hook))


def assert_hooks(records: HookRecords, surface: str, call_id: str, expected: list[str]) -> None:
    settled: Final = held(
        lambda: records.for_call(call_id),
        lambda found: probe_hooks(records, surface, found) == expected,
        holding=3,
        seconds=20,
    )
    assert {record["call_type"] for record in settled} == {SURFACE_CALL_TYPES[surface]}, settled
