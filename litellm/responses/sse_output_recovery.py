"""
Shared helpers for recovering Responses API output items from raw SSE chunks.

The same recovery logic is needed in multiple places (e.g. the ChatGPT
Responses transformation and the LiteLLM Responses-to-Chat-Completions
bridge). Keep the implementation in a single module so a fix in one
caller automatically applies to all of them.
"""

import json
from typing import Any, Dict, Optional

from litellm.constants import STREAM_SSE_DONE_STRING

_MAX_CONTENT_INDEX = 1024


def parse_sse_json_chunk(chunk: str) -> Optional[Dict[str, Any]]:
    """Parse a single raw SSE line into a JSON object dict.

    Returns ``None`` for empty lines, ``event:`` lines, ``[DONE]`` markers,
    invalid JSON, or non-dict payloads. Centralizes the parsing step that
    feeds into the recovery helpers in this module so behavior stays
    consistent across all callers.
    """
    # Import locally to avoid a circular import with the streaming handler.
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

    stripped_chunk = (CustomStreamWrapper._strip_sse_data_from_chunk(chunk.strip()) or "").strip()
    if not stripped_chunk or stripped_chunk == STREAM_SSE_DONE_STRING or stripped_chunk.startswith("event:"):
        return None
    try:
        parsed_chunk = json.loads(stripped_chunk)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed_chunk, dict):
        return None
    return parsed_chunk


def record_output_item_chunk(
    parsed_chunk: Dict[str, Any],
    output_items: Dict[int, Dict[str, Any]],
) -> None:
    """Record an OUTPUT_ITEM_DONE chunk into ``output_items`` keyed by
    ``output_index`` (falling back to the next free slot when missing).
    """
    item = parsed_chunk.get("item")
    if not isinstance(item, dict):
        return
    try:
        output_index_raw = parsed_chunk.get("output_index")
        if output_index_raw is None:
            raise ValueError("missing output_index")
        output_index = int(output_index_raw)
    except (TypeError, ValueError):
        output_index = len(output_items)
    output_items[output_index] = item


def record_output_text_chunk(
    parsed_chunk: Dict[str, Any],
    output_items: Dict[int, Dict[str, Any]],
    text_only_items: Dict[int, Dict[str, Any]],
) -> None:
    """Record an OUTPUT_TEXT_DONE chunk as a synthetic message item in
    ``text_only_items``. Real OUTPUT_ITEM_DONE events already captured in
    ``output_items`` take precedence at the same ``output_index``.
    """
    text = parsed_chunk.get("text")
    if not isinstance(text, str):
        return

    try:
        output_index_raw = parsed_chunk.get("output_index")
        if output_index_raw is None:
            raise ValueError("missing output_index")
        output_index = int(output_index_raw)
    except (TypeError, ValueError):
        output_index = len(text_only_items)

    if output_index in output_items:
        return

    item = text_only_items.get(output_index)
    if item is None:
        item = {
            "type": "message",
            "id": parsed_chunk.get("item_id") or f"msg_{output_index}",
            "role": "assistant",
            "status": "completed",
            "content": [],
        }
        text_only_items[output_index] = item

    content = item.setdefault("content", [])
    if not isinstance(content, list):
        return

    try:
        content_index_raw = parsed_chunk.get("content_index")
        if content_index_raw is None:
            raise ValueError("missing content_index")
        content_index = int(content_index_raw)
    except (TypeError, ValueError):
        content_index = len(content)

    if content_index < 0 or content_index > _MAX_CONTENT_INDEX:
        return

    while len(content) <= content_index:
        content.append(
            {
                "type": "output_text",
                "text": "",
                "annotations": [],
            }
        )

    content_item = content[content_index]
    if not isinstance(content_item, dict):
        content_item = {}
        content[content_index] = content_item

    content_item["type"] = "output_text"
    content_item["text"] = text
    if parsed_chunk.get("annotations") is not None:
        content_item["annotations"] = parsed_chunk["annotations"]
    else:
        content_item.setdefault("annotations", [])
