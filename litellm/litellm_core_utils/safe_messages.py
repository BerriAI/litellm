"""
Safe traversal of message and request content for guardrails.

Chat-completion `message["content"]` may be either a plain string OR a list of
content parts (multimodal: `[{"type": "text", "text": "..."}, {"type": "image_url", ...}]`).
Guardrails that only check `isinstance(content, str)` silently skip the list
shape, allowing users to bypass moderation, secret detection, and banned-keyword
checks. All guardrail and moderation hooks must traverse content through these
helpers so the list shape is handled consistently in one place.
"""

from typing import Iterator, List, Optional, Tuple

# (msg_idx, part_idx_or_None, text)
TextSlot = Tuple[int, Optional[int], str]
# (source, slot, text) where source ∈ {"messages", "prompt", "input"}
RequestTextSlot = Tuple[str, TextSlot, str]


def iter_message_texts(messages: List[dict]) -> Iterator[TextSlot]:
    """Yield (msg_idx, part_idx, text) for every text fragment in `messages`.

    - `part_idx` is None when the message's `content` is a plain string.
    - `part_idx` is an int when `content` is a list; only items whose
      `type == "text"` (or bare-string items, which bedrock allows) are yielded.
    - Messages with content that is None, missing, or an unexpected shape are
      skipped silently.
    - Non-string `text` values inside a part dict are skipped silently.
    Does not mutate `messages`.
    """
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            yield msg_idx, None, content
        elif isinstance(content, list):
            for part_idx, part in enumerate(content):
                if isinstance(part, str):
                    yield msg_idx, part_idx, part
                elif isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        yield msg_idx, part_idx, text


def set_message_text(
    messages: List[dict],
    msg_idx: int,
    part_idx: Optional[int],
    new_text: str,
) -> None:
    """Write `new_text` into the slot previously yielded by `iter_message_texts`.

    - If `part_idx is None`, replaces the message's string `content`.
    - If `part_idx is not None`, replaces `content[part_idx]["text"]`, or replaces
      a bare-string list entry directly.
    - No-ops if the slot no longer exists (defensive against concurrent mutation).
    """
    if msg_idx < 0 or msg_idx >= len(messages):
        return
    msg = messages[msg_idx]
    if not isinstance(msg, dict):
        return
    if part_idx is None:
        msg["content"] = new_text
        return
    content = msg.get("content")
    if not isinstance(content, list) or part_idx < 0 or part_idx >= len(content):
        return
    part = content[part_idx]
    if isinstance(part, str):
        content[part_idx] = new_text
    elif isinstance(part, dict) and part.get("type") == "text":
        part["text"] = new_text


def collect_message_text(messages: List[dict], separator: str = "") -> str:
    """Concatenate every text fragment from `messages` in iteration order.

    Convenience for moderation hooks that batch one string per request.
    Default separator is "" to preserve existing concatenation semantics.
    """
    return separator.join(text for _, _, text in iter_message_texts(messages))


def iter_request_texts(data: dict) -> Iterator[RequestTextSlot]:
    """Yield (source, slot, text) for every scannable text in a proxy request.

    Walks `data["messages"]`, `data["prompt"]`, and `data["input"]`. Each yielded
    `slot` is a `TextSlot` that can be passed to `set_request_text` to write a
    redacted value back into the same location.

    For `prompt` and `input` (which are str or list[str]), the slot's
    `part_idx` is None for the string shape and an int for the list shape.
    """
    messages = data.get("messages")
    if isinstance(messages, list):
        for msg_idx, part_idx, text in iter_message_texts(messages):
            yield "messages", (msg_idx, part_idx, text), text

    for source in ("prompt", "input"):
        value = data.get(source)
        if isinstance(value, str):
            yield source, (0, None, value), value
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, str):
                    yield source, (0, idx, item), item


def set_request_text(data: dict, source: str, slot: TextSlot, new_text: str) -> None:
    """Mutating counterpart to `iter_request_texts`.

    Writes `new_text` back into the slot that was yielded by `iter_request_texts`.
    """
    msg_idx, part_idx, _ = slot
    if source == "messages":
        messages = data.get("messages")
        if isinstance(messages, list):
            set_message_text(messages, msg_idx, part_idx, new_text)
        return
    if source not in ("prompt", "input"):
        return
    value = data.get(source)
    if part_idx is None:
        if isinstance(value, str):
            data[source] = new_text
        return
    if isinstance(value, list) and 0 <= part_idx < len(value):
        if isinstance(value[part_idx], str):
            value[part_idx] = new_text
