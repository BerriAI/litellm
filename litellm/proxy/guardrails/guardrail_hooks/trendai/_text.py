# Derived from tm-v1-ai-guard-litellm-plugin revision 6cafc143f62962a98d4eb5abe9f608c61ff194d4.
# This file has been modified for integration into LiteLLM.

from collections.abc import Iterator, Sequence
from itertools import chain
from typing import Final

from litellm.llms.base_llm.guardrail_translation.utils import message_slot_texts, message_text_slot_count
from litellm.types.llms.openai import AllMessageValues

from ._models import TrendAIRequestPrompt, TrendAITextWindow


def utf8_windows(content: str, *, chunk_size_bytes: int, overlap_chars: int) -> tuple[TrendAITextWindow, ...]:
    """Split ``content`` into windows of at most ``chunk_size_bytes`` UTF-8 bytes.

    Consecutive windows share their last ``overlap_chars`` characters so a finding that straddles a
    window boundary is still seen whole by one scan. The overlap is capped below the window length so
    every window makes forward progress.
    """
    return tuple(_iter_utf8_windows(content, chunk_size_bytes, overlap_chars))


def _iter_utf8_windows(content: str, chunk_size_bytes: int, overlap_chars: int) -> Iterator[TrendAITextWindow]:
    encoded: Final = content.encode("utf-8")
    byte_offset = 0  # rebind-ok: window cursor advances across the loop
    char_offset = 0  # rebind-ok: window cursor advances across the loop
    while byte_offset < len(encoded):
        text = (
            encoded[byte_offset : byte_offset + chunk_size_bytes].decode("utf-8", errors="ignore")
            or content[char_offset]
        )
        yield TrendAITextWindow(text=text, start=char_offset)
        end_byte_offset = byte_offset + len(text.encode("utf-8"))
        if end_byte_offset >= len(encoded):
            return
        overlap = min(max(overlap_chars, 0), len(text) - 1)
        byte_offset = end_byte_offset - len(text[len(text) - overlap :].encode("utf-8"))
        char_offset += len(text) - overlap


def merge_redaction(original: str, current: str, redacted: str) -> str | None:
    """Apply the masks ``redacted`` adds over ``original`` on top of the masks already in ``current``.

    Returns None when the three texts do not line up character for character, since a positional
    merge would then scramble the output.
    """
    if len(original) != len(current) or len(redacted) != len(original):
        return None
    return "".join(
        redacted_char if redacted_char != original_char else current_char
        for original_char, current_char, redacted_char in zip(original, current, redacted, strict=True)
    )


def apply_window_redaction(content: str, window: TrendAITextWindow, redacted: str) -> str | None:
    end: Final = window.start + len(window.text)
    merged: Final = merge_redaction(window.text, content[window.start : end], redacted)
    if merged is None:
        return None
    return f"{content[: window.start]}{merged}{content[end:]}"


def locate_request_prompt(
    texts: Sequence[str],
    structured_messages: Sequence[AllMessageValues] | None,
) -> TrendAIRequestPrompt | None:
    """Locate the last user turn by message order, never by matching its text.

    A different flat-text layout can still be scanned, but cannot be safely rewritten.
    """
    messages: Final = tuple(structured_messages or ())
    user_index: Final = next(
        (index for index in reversed(range(len(messages))) if messages[index]["role"] == "user"), None
    )
    if messages and user_index is None:
        return None
    parts: Final = message_slot_texts(messages[user_index]) if user_index is not None else tuple(texts)
    prompt: Final = "".join(parts).strip()
    if not prompt:
        return None
    start: Final = sum(map(message_text_slot_count, messages[:user_index])) if user_index is not None else 0
    normalized_texts: Final = (
        tuple(chain.from_iterable(map(message_slot_texts, messages))) if messages else tuple(texts)
    )
    text_range: Final = (start, start + len(parts)) if normalized_texts == tuple(texts) else None
    return TrendAIRequestPrompt(prompt=prompt, text_range=text_range)


def redact_request_texts(texts: Sequence[str], prompt: TrendAIRequestPrompt, redacted: str) -> tuple[str, ...] | None:
    """Write a redacted prompt back over the ``texts`` entries the prompt was assembled from.

    A single-part prompt is replaced outright. A multi-part prompt whose redaction kept its length
    is split back into the original parts; otherwise the first part carries the whole redaction and
    the rest are blanked, so no original part can leak.
    """
    if prompt.text_range is None:
        return None
    start, end = prompt.text_range
    parts: Final = tuple(texts[start:end])
    if len(parts) == 1:
        return (*texts[:start], redacted, *texts[end:])
    joined: Final = "".join(parts)
    leading: Final = len(joined) - len(joined.lstrip())
    aligned: Final = f"{joined[:leading]}{redacted}{joined[len(joined.rstrip()) :]}"
    if len(aligned) != len(joined):
        return (*texts[:start], redacted, *("" for _ in parts[1:]), *texts[end:])
    offsets: Final = tuple(sum(len(part) for part in parts[:index]) for index in range(len(parts) + 1))
    split: Final = tuple(aligned[offsets[index] : offsets[index + 1]] for index in range(len(parts)))
    return (*texts[:start], *split, *texts[end:])
