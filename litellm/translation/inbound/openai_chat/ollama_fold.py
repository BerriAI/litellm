"""The ollama NDJSON wire-chunk fold (wave-3).

Pure helpers that turn one ollama ``/api/chat`` wire chunk into the chat-
completion chunk bodies v1's CustomStreamWrapper emits for that family. The
stateful orchestration (the two reasoning flags ride ``StreamState``) stays in
``stream._step_ollama_chat``; everything here is pure over primitives so the
fold lives beside its dialect without growing the dialect file past the cap.
"""

from __future__ import annotations

from ...ir import Body, PlainJson


def ollama_reasoning_split(
    thinking: PlainJson, content: PlainJson, started: bool, finished: bool
) -> tuple[str | None, str | None, bool, bool]:
    """v1's two-flag think-tag machine. A ``thinking`` field sets started;
    truthy content FIRST flips finished when reasoning had started (only the
    tag-bearing chunk is reasoning), then ``<think>``/``</think>`` tags are
    stripped and flip the flags, and the leftover text rides reasoning while
    started-and-not-finished, else content. A single chunk carrying both an
    open and a close tag loses the reasoning entirely (v1 parity)."""
    reasoning_out: str | None = None
    content_out: str | None = None
    if isinstance(thinking, str) and thinking:
        reasoning_out = thinking
        started = True
    if not (isinstance(content, str) and content):
        return reasoning_out, content_out, started, finished
    if started and not finished:
        finished = True
    text, started, finished = _strip_think_tags(content, started, finished)
    if started and not finished:
        reasoning_out = text
    else:
        content_out = text
    return reasoning_out, content_out, started, finished


def _strip_think_tags(
    text: str, started: bool, finished: bool
) -> tuple[str, bool, bool]:
    if "<think>" in text:
        text = text.replace("<think>", "")
        started = True
    if "</think>" in text and started:
        text = text.replace("</think>", "")
        finished = True
    return text, started, finished


def ollama_delta_body(
    sent_role: bool,
    model: str,
    content_out: str | None,
    reasoning_out: str | None,
    entries: tuple[PlainJson, ...] | None,
) -> Body | None:
    """The content/reasoning/tool delta chunk, or None when the wire chunk
    bears no payload (v1's empty-chunk drop)."""
    if content_out is None and reasoning_out is None and entries is None:
        return None
    delta: dict[str, PlainJson] = {
        "role": None if sent_role else "assistant",
        "content": content_out,
        "provider_specific_fields": None,
    }
    if reasoning_out is not None:
        delta = {**delta, "reasoning_content": reasoning_out}
    if entries is not None:
        delta = {**delta, "tool_calls": list(entries)}
    return {
        "model": model,
        "object": "chat.completion.chunk",
        "system_fingerprint": None,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


def ollama_bodies(
    model: str, delta: Body | None, finish: PlainJson, usage: PlainJson
) -> tuple[Body, ...]:
    bodies: tuple[Body, ...] = () if delta is None else (delta,)
    if finish is not None:
        # v1's wrapper-synthesized SPLIT flush (a payload-bearing done chunk)
        # carries no system_fingerprint; a standalone finish is the iterator's
        # own chunk and does (probed)
        final: Body = {
            "model": model,
            "object": "chat.completion.chunk",
            **({} if delta is not None else {"system_fingerprint": None}),
            "choices": [
                {"index": 0, "delta": {"content": None}, "finish_reason": finish}
            ],
        }
        bodies = (*bodies, final)
    if isinstance(usage, dict):
        bodies = (
            *bodies,
            {
                "model": model,
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": usage,
            },
        )
    return bodies
