import json
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

from integration._support.anthropic_sse import message_stream
from integration._support.openai_wire import chat_reply
from integration._support.upstream import aws_event_stream_frame
from integration._support.wire import Reply, Request
from integration.providers._api_base_trailing_slash_support import VECTOR, marker_of, reply_text, request_json
from pydantic import JsonValue

COUNTED_TOKENS: Final = 7
RELEVANCE: Final = 0.75
_USAGE: Final = {"input_tokens": 3, "output_tokens": 2}


def _json(payload: Mapping[str, JsonValue]) -> Reply:
    return Reply(body=json.dumps(payload).encode())


def _path(request: Request) -> str:
    return urlsplit(request.target).path


def _wants_stream(request: Request) -> bool:
    return request_json(request).get("stream") is True


def anthropic_message(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{marker}",
        "type": "message",
        "role": "assistant",
        "model": "scripted",
        "content": [{"type": "text", "text": reply_text(marker)}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": dict(_USAGE),
    }


def _anthropic(request: Request) -> Reply:
    if _path(request).endswith("/count_tokens"):
        return _json({"input_tokens": COUNTED_TOKENS})
    marker: Final = marker_of(request)
    if _wants_stream(request):
        return Reply(
            content_type="text/event-stream", chunks=message_stream(f"msg_{marker}", "scripted", reply_text(marker))
        )
    return _json(anthropic_message(marker))


def _gemini_candidate(text: str, finished: bool) -> dict[str, JsonValue]:
    finish: Final[dict[str, JsonValue]] = {"finishReason": "STOP"} if finished else {}
    return {"content": {"role": "model", "parts": [{"text": text}]}, "index": 0, **finish}


def _gemini_frame(text: str, finished: bool) -> bytes:
    usage: Final = {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}
    frame: Final = {"candidates": [_gemini_candidate(text, finished)], "usageMetadata": usage}
    return b"data: " + json.dumps(frame).encode() + b"\r\n\r\n"


def _gemini(request: Request) -> Reply:
    path: Final = _path(request)
    marker: Final = marker_of(request)
    if path.endswith(":batchEmbedContents"):
        return _json({"embeddings": [{"values": list(VECTOR)}]})
    if path.endswith(":streamGenerateContent"):
        return Reply(
            content_type="text/event-stream",
            chunks=(_gemini_frame("reply ", False), _gemini_frame(marker, True)),
        )
    return _json(
        {
            "candidates": [_gemini_candidate(reply_text(marker), True)],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
        }
    )


def _embedding_list(model: str) -> Reply:
    return _json(
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": list(VECTOR)}],
            "model": model,
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }
    )


def _openai_like(model: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        path: Final = _path(request)
        marker: Final = marker_of(request)
        if path.endswith("/embeddings"):
            return _embedding_list(model)
        if path.endswith("/audio/transcriptions"):
            return _json({"text": reply_text(marker), "model": model})
        return chat_reply(f"chatcmpl-{marker}", model, reply_text(marker), stream=_wants_stream(request))

    return respond


def _converse_event(kind: str, payload: Mapping[str, JsonValue]) -> bytes:
    headers: Final = {":event-type": kind, ":content-type": "application/json", ":message-type": "event"}
    return aws_event_stream_frame(headers, json.dumps(payload, separators=(",", ":")).encode())


def _bedrock_converse(request: Request) -> Reply:
    marker: Final = marker_of(request)
    usage: Final[dict[str, JsonValue]] = {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5}
    if _path(request).endswith("/converse-stream"):
        events: Final[tuple[tuple[str, Mapping[str, JsonValue]], ...]] = (
            ("messageStart", {"role": "assistant"}),
            ("contentBlockDelta", {"delta": {"text": "reply "}, "contentBlockIndex": 0}),
            ("contentBlockDelta", {"delta": {"text": marker}, "contentBlockIndex": 0}),
            ("contentBlockStop", {"contentBlockIndex": 0}),
            ("messageStop", {"stopReason": "end_turn"}),
            ("metadata", {"usage": usage, "metrics": {"latencyMs": 1}}),
        )
        return Reply(
            content_type="application/vnd.amazon.eventstream",
            chunks=tuple(_converse_event(kind, payload) for kind, payload in events),
        )
    return _json(
        {
            "output": {"message": {"role": "assistant", "content": [{"text": reply_text(marker)}]}},
            "stopReason": "end_turn",
            "usage": usage,
            "metrics": {"latencyMs": 1},
        }
    )


def _cohere_rerank(request: Request) -> Reply:
    return _json(
        {
            "id": f"rerank-{marker_of(request)}",
            "results": [{"index": 0, "relevance_score": RELEVANCE}],
            "meta": {"billed_units": {"search_units": 1}},
        }
    )


_BY_PREFIX: Final[Mapping[str, Callable[[Request], Reply]]] = MappingProxyType(
    {
        "an": _anthropic,
        "bi": lambda request: _json(anthropic_message(marker_of(request))),
        "bc": _bedrock_converse,
        "gm": _gemini,
        "mi": _openai_like("mistral-small-latest"),
        "db": _openai_like("databricks-meta-llama-3-3-70b-instruct"),
        "co": _cohere_rerank,
    }
)


def join_peer(request: Request) -> Reply:
    return _BY_PREFIX[request.target.split("/")[1]](request)
