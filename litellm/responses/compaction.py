"""
Universal server-side context compaction for the Responses API.

Summarizes conversation history when input tokens exceed a configured threshold,
using the same model the caller is already using. Works across all providers.
"""

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

import litellm
from litellm._logging import verbose_logger

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a conversation summarizer. The user will provide a transcript of a "
    "conversation between a user and an assistant. Produce a concise summary that "
    "captures: (1) what the user asked or talked about, (2) what the assistant "
    "replied, and (3) any decisions, conclusions, or pending questions. "
    "Write the summary in plain prose, in the third person (e.g. 'The user asked "
    "about X. The assistant explained Y.'). Do NOT use XML tags, markdown headers, "
    "or any special formatting. "
    "Act like a compression algorithm: use as few words as possible while preserving "
    "all important information. For simple or repetitive conversations, a few "
    "sentences may suffice. For complex conversations involving multiple disparate "
    "tasks, detailed technical decisions, or extensive context, you may use more "
    "detail. The absolute maximum is 5000 words, but most summaries should be "
    "significantly shorter than that."
)

MAX_SUMMARY_TOKENS = 4096

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
            verbose_logger.warning(
                "compaction: context_management has type='compaction' but no "
                "compact_threshold — compaction will not trigger"
            )
            return None
    return None


def _extract_summary(text: str) -> str:
    """Extract the summary, stripping any XML-style tags the model may have added."""
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
    litellm_metadata: Optional[Dict[str, Any]] = None,
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

    char_count = sum(
        len(str(msg.get("content", ""))) for msg in messages
    )
    token_count = char_count // 4
    verbose_logger.debug(
        "compaction: estimated token_count=%d, threshold=%d", token_count, threshold
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
        {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]

    acompletion_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": summarization_messages,
        "max_tokens": MAX_SUMMARY_TOKENS,
    }
    if custom_llm_provider is not None:
        acompletion_kwargs["custom_llm_provider"] = custom_llm_provider
    if litellm_metadata is not None:
        acompletion_kwargs["metadata"] = copy.deepcopy(litellm_metadata)

    summary_response = await litellm.acompletion(**acompletion_kwargs)

    summary_text_raw = summary_response.choices[0].message.content or ""
    summary_text = _extract_summary(summary_text_raw)

    verbose_logger.debug("compaction: summary length=%d chars", len(summary_text))

    compacted_messages: List[Dict[str, Any]] = [
        {"role": "user", "content": summary_text},
    ]

    return compacted_messages, summary_text
