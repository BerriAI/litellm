"""
Universal server-side context compaction for the Responses API.

Summarizes conversation history when input tokens exceed a configured threshold,
using the same model the caller is already using. Works across all providers.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import litellm
from litellm._logging import verbose_logger

SUMMARIZATION_SYSTEM_PROMPT = (
    "You have written a partial transcript for the initial task above. "
    "Please write a summary of the transcript. The purpose of this summary is "
    "to provide continuity so you can continue to make progress towards solving "
    "the task in a future context, where the raw history above may not be "
    "accessible and will be replaced with this summary. Write down anything "
    "that would be helpful, including the state, next steps, learnings etc. "
    "You must wrap your summary in a <summary></summary> block."
)

MIN_COMPACT_THRESHOLD = 1000


def _get_compact_threshold(context_management: List[Dict[str, Any]]) -> Optional[int]:
    """Extract the compact_threshold from a context_management list, if present."""
    for entry in context_management:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == "compaction":
            threshold = entry.get("compact_threshold")
            if threshold is not None:
                return max(int(threshold), MIN_COMPACT_THRESHOLD)
            return None
    return None


def _extract_summary(text: str) -> str:
    """Pull content out of <summary>...</summary> tags, falling back to the full text."""
    match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _serialize_messages_for_summary(messages: List[Dict[str, Any]]) -> str:
    """Render a message list into a readable transcript for the summarizer."""
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    text_parts.append(block.get("text", str(block)))
                else:
                    text_parts.append(str(block))
            content = "\n".join(text_parts)
        parts.append(f"[{role}]: {content}")
    return "\n\n".join(parts)


async def maybe_compact_context(
    messages: List[Dict[str, Any]],
    model: str,
    context_management: List[Dict[str, Any]],
    custom_llm_provider: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Check whether compaction should trigger and, if so, summarize ALL messages.

    Returns:
        (messages, summary_text):
        - If compaction triggered: messages is [{role: "user", content: summary}],
          summary_text is the raw summary string.
        - If not triggered: messages is unchanged, summary_text is None.
    """
    threshold = _get_compact_threshold(context_management)
    if threshold is None:
        return messages, None

    token_count = litellm.token_counter(model=model, messages=messages)
    verbose_logger.debug(
        "compaction: token_count=%d, threshold=%d", token_count, threshold
    )

    if token_count <= threshold:
        return messages, None

    verbose_logger.info(
        "compaction: triggering summarization (tokens=%d > threshold=%d)",
        token_count,
        threshold,
    )

    transcript = _serialize_messages_for_summary(messages)

    summarization_messages: List[Dict[str, Any]] = [
        {"role": "user", "content": transcript},
        {"role": "user", "content": SUMMARIZATION_SYSTEM_PROMPT},
    ]

    acompletion_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": summarization_messages,
    }
    if custom_llm_provider is not None:
        acompletion_kwargs["custom_llm_provider"] = custom_llm_provider

    summary_response = await litellm.acompletion(**acompletion_kwargs)

    summary_text_raw = summary_response.choices[0].message.content or ""
    summary_text = _extract_summary(summary_text_raw)

    verbose_logger.debug("compaction: summary length=%d chars", len(summary_text))

    compacted_messages: List[Dict[str, Any]] = [
        {"role": "user", "content": summary_text},
    ]

    return compacted_messages, summary_text
