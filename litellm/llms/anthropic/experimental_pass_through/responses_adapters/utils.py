"""
Shared utilities for Anthropic <-> OpenAI Responses API web search translation.

Used by both the streaming (streaming_iterator.py) and non-streaming
(transformation.py) response paths.
"""

from typing import Any, Dict, List, Tuple

from litellm.types.llms.anthropic import AnthropicResponseContentBlockText


def build_web_tool_use(item: Any) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Build server_tool_use content block from a response web_search_call item.

    Returns (server_tool_use_block, input_dict) where:
    - server_tool_use_block: {"type": "server_tool_use", "id": ..., "name": "web_search"|"web_fetch"}
    - input_dict: {"query": "..."} for search, {"url": "..."} for open_page/find_in_page
    """
    if not isinstance(item, dict):
        item = item.model_dump()
    action = item.get("action", {})
    action_type = action.get("type", "search")
    if action_type in ("open_page", "find_in_page"):
        name = "web_fetch"
        input_dict = {"url": action.get("url", "")}
    else:
        name = "web_search"
        queries = action.get("queries", {})
        if queries:
            query = "\n".join(queries)
        else:
            query = action.get("query", "")
        input_dict = {"query": query}
    block = {
        "type": "server_tool_use",
        "name": name,
        "id": item.get("id", ""),
    }
    return block, input_dict


def build_web_search_results_from_annotations(
    web_tool_uses: List[Dict[str, Any]],
    annotations: list,
) -> Tuple[List[Dict[str, Any]], List[tuple]]:
    """Build web search content blocks and citations from search calls and annotations.

    Returns (content_blocks, citations) where:
    - content_blocks: web_search_tool_result blocks
    - citations: list of (start_index, end_index, citation_dict) tuples sorted by position
    """
    citations: List[tuple] = []
    seen_urls: Dict[str, Dict[str, Any]] = {}

    for ann in annotations:
        if not isinstance(ann, dict):
            ann = ann.model_dump()
        if ann.get("type") != "url_citation":
            continue

        url = ann.get("url", "")
        title = ann.get("title", "")
        start = ann.get("start_index", 0) or 0
        end = ann.get("end_index", 0) or 0

        citations.append((start, end, {
            "type": "web_search_result_location",
            "url": url,
            "title": title,
            "cited_text": title,
        }))

        if url and url not in seen_urls:
            seen_urls[url] = {
                "type": "web_search_result",
                "url": url,
                "title": title,
            }

    citations.sort(key=lambda x: x[0])
    search_results = list(seen_urls.values())
    search_calls = [c for c in web_tool_uses if c["name"] == "web_search"]

    content_blocks: List[Dict[str, Any]] = []
    if search_calls:
        content_blocks.append(
            {
                "type": "web_search_tool_result",
                "tool_use_id": search_calls[0]["id"],
                "content": search_results,
            }
        )

    return content_blocks, citations


def build_text_blocks_with_citations(
    text: str,
    citations: List[tuple],
) -> List[Dict[str, Any]]:
    """Split text into alternating uncited / cited Anthropic text blocks.

    Each citation tuple is (start_index, end_index, citation_dict).
    text[start:end] is the cited range; everything else is uncited.
    """
    if not citations:
        return [
            AnthropicResponseContentBlockText(type="text", text=text).model_dump()
        ]

    blocks: List[Dict[str, Any]] = []
    pos = 0

    for start, end, citation in citations:
        if pos < start:
            blocks.append(
                AnthropicResponseContentBlockText(
                    type="text", text=text[pos:start]
                ).model_dump()
            )
        if start < end:
            block = AnthropicResponseContentBlockText(
                type="text", text=text[start:end]
            ).model_dump()
            block["citations"] = [citation]
            blocks.append(block)
        pos = end

    if pos < len(text):
        blocks.append(
            AnthropicResponseContentBlockText(
                type="text", text=text[pos:]
            ).model_dump()
        )

    return blocks
