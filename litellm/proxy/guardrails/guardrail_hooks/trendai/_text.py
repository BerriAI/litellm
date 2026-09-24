# Derived from tm-v1-ai-guard-litellm-plugin revision 6cafc143f62962a98d4eb5abe9f608c61ff194d4.
# This file has been modified for integration into LiteLLM.
# Licensed under the Apache License, Version 2.0. See LICENSE.txt in this directory.

from collections.abc import Iterator, Sequence
from typing import Final

from litellm.llms.base_llm.guardrail_translation.utils import message_slot_texts
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


def _last_user_text_parts(structured_messages: Sequence[AllMessageValues]) -> tuple[str, ...] | None:
    last_user_message: Final = next(
        (message for message in reversed(structured_messages) if message.get("role") == "user"), None
    )
    return message_slot_texts(last_user_message) if last_user_message is not None else None


def _last_occurrence(texts: Sequence[str], parts: Sequence[str]) -> int | None:
    return next(
        (
            start
            for start in range(len(texts) - len(parts), -1, -1)
            if tuple(texts[start : start + len(parts)]) == tuple(parts)
        ),
        None,
    )


def locate_request_prompt(
    texts: Sequence[str],
    structured_messages: Sequence[AllMessageValues] | None,
) -> TrendAIRequestPrompt | None:
    """Pick the text Trend AI Guard scans for a request and where it lives in ``texts``.

    The scanned prompt is the last user turn, matching the upstream plugin. Its text parts are
    located as a run inside ``texts`` so a redacted prompt can be written back to exactly those
    entries. Without structured messages the whole ``texts`` list is the prompt.
    """
    parts: Final = _last_user_text_parts(structured_messages) if structured_messages else tuple(texts)
    if parts is None or not parts:
        return None
    prompt: Final = "".join(parts).strip()
    if not prompt:
        return None
    start: Final = _last_occurrence(texts, parts)
    if start is None:
        return None
    return TrendAIRequestPrompt(prompt=prompt, start=start, end=start + len(parts))


def redact_request_texts(texts: Sequence[str], prompt: TrendAIRequestPrompt, redacted: str) -> tuple[str, ...]:
    """Write a redacted prompt back over the ``texts`` entries the prompt was assembled from.

    A single-part prompt is replaced outright. A multi-part prompt whose redaction kept its length
    is split back into the original parts; otherwise the first part carries the whole redaction and
    the rest are blanked, so no original part can leak.
    """
    parts: Final = tuple(texts[prompt.start : prompt.end])
    if len(parts) == 1:
        return (*texts[: prompt.start], redacted, *texts[prompt.end :])
    joined: Final = "".join(parts)
    leading: Final = len(joined) - len(joined.lstrip())
    aligned: Final = f"{joined[:leading]}{redacted}{joined[len(joined.rstrip()) :]}"
    if len(aligned) != len(joined):
        return (*texts[: prompt.start], redacted, *("" for _ in parts[1:]), *texts[prompt.end :])
    offsets: Final = tuple(sum(len(part) for part in parts[:index]) for index in range(len(parts) + 1))
    split: Final = tuple(aligned[offsets[index] : offsets[index + 1]] for index in range(len(parts)))
    return (*texts[: prompt.start], *split, *texts[prompt.end :])
