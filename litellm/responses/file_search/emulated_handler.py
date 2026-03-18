"""
Emulated file_search for providers that don't support the tool natively.

Flow:
  1. Convert file_search tools to a single function tool definition.
  2. Call the provider with the function tool.
  3. If the provider issues a file_search function_call, execute vector search
     via litellm.vector_stores.main.asearch().
  4. Feed results back and get the final answer.
  5. Wrap everything in OpenAI Responses-API format:
       [file_search_call output item] + [message output item with file_citation annotations]
"""

import json
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

from litellm._logging import verbose_logger
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.vector_stores import VectorStoreSearchResult

# Keep ToolParam broad so we stay compatible with both dict and Pydantic forms
ToolParam = Any

FILE_SEARCH_FUNCTION_NAME = "litellm_file_search"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def should_use_emulated_file_search(
    tools: Optional[Iterable[ToolParam]],
    provider_config: Any,  # BaseResponsesAPIConfig
) -> bool:
    """Return True when there is a file_search tool and the provider can't handle it natively."""
    if not tools:
        return False
    has_fs = any(
        isinstance(t, dict) and t.get("type") == "file_search" for t in tools
    )
    if not has_fs:
        return False
    return provider_config is None or not provider_config.supports_native_file_search()


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------

def _build_function_tool(vector_store_ids: List[str]) -> Dict[str, Any]:
    """
    Create a Responses API function-tool definition that describes file search.
    The function accepts one or more natural-language queries (like OpenAI's native
    file_search); LiteLLM runs the actual vector search against the configured
    vector stores.

    Note: Uses Responses API format (name/description/parameters at top level),
    NOT Chat Completion format (nested under "function"), so that the
    LiteLLMCompletionResponsesConfig transformation picks up name and description.
    """
    return {
        "type": "function",
        "name": FILE_SEARCH_FUNCTION_NAME,
        "description": (
            "Search the knowledge base for information relevant to the query. "
            "Use this whenever you need to look up specific facts, documents, "
            "or content from the vector store. You can provide multiple queries "
            "to search for different aspects of the information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "One or more search queries to look up in the vector store. "
                        "Multiple queries help find comprehensive information from "
                        "different angles."
                    ),
                },
                "vector_store_id": {
                    "type": "string",
                    "description": "ID of the vector store to search.",
                    "enum": vector_store_ids,
                },
            },
            "required": ["queries"],
        },
    }


def _replace_file_search_tools(
    tools: Optional[Iterable[ToolParam]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Replace all file_search tools with a single function tool.

    Returns:
        (new_tools_list, all_vector_store_ids)
    """
    non_file_search: List[Dict[str, Any]] = []
    vector_store_ids: List[str] = []

    for tool in (tools or []):
        if isinstance(tool, dict) and tool.get("type") == "file_search":
            ids = tool.get("vector_store_ids") or []
            vector_store_ids.extend(ids)
        else:
            non_file_search.append(tool)

    # Deduplicate while preserving order
    unique_ids: List[str] = list(dict.fromkeys(vector_store_ids))
    if unique_ids:
        non_file_search.append(_build_function_tool(unique_ids))

    return non_file_search, unique_ids


# ---------------------------------------------------------------------------
# Search execution
# ---------------------------------------------------------------------------

async def _run_vector_searches(
    queries: List[str],
    vector_store_ids: List[str],
) -> Tuple[List[str], List[VectorStoreSearchResult]]:
    """
    Run `asearch` against all vector stores for all queries and collect results.

    Args:
        queries: List of search queries to execute (like OpenAI's multi-query approach)
        vector_store_ids: Vector store IDs to search

    Returns:
        (queries_list, combined_results)
    """
    import litellm.vector_stores.main as vs_main

    all_results: List[VectorStoreSearchResult] = []
    ids_to_search = vector_store_ids

    # Execute each query against all vector stores
    for query in queries:
        for vs_id in ids_to_search:
            try:
                response = await vs_main.asearch(
                    vector_store_id=vs_id,
                    query=query,
                )
                results_data = response.get("data") if isinstance(response, dict) else getattr(response, "data", None)
                if results_data:
                    all_results.extend(results_data)
            except Exception as exc:
                verbose_logger.warning(
                    "file_search emulated: search failed for query='%s', vector_store_id='%s': %s",
                    query,
                    vs_id,
                    exc,
                )

    return queries, all_results


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _get_field(result: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict/TypedDict or an attribute-based object."""
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _format_search_results_as_tool_output(
    results: List[VectorStoreSearchResult],
) -> str:
    """Serialize search results into a string to pass back as the tool's output."""
    if not results:
        return "No results found in the vector store."

    parts: List[str] = []
    for i, result in enumerate(results, 1):
        score = _get_field(result, "score")
        file_id = _get_field(result, "file_id")
        filename = _get_field(result, "filename")
        content_items = _get_field(result, "content") or []
        text_chunks = [
            c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
            for c in content_items
        ]
        text = " ".join(t for t in text_chunks if t)

        header = f"[Result {i}"
        if filename:
            header += f" | {filename}"
        if file_id:
            header += f" | file_id={file_id}"
        if score is not None:
            header += f" | score={score:.3f}"
        header += "]"

        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


def _build_search_results_for_include(
    results: List[VectorStoreSearchResult],
) -> List[Dict[str, Any]]:
    """
    Convert VectorStoreSearchResult objects to the format expected in
    file_search_call.search_results (mirrors OpenAI's include= format).

    All chunks are returned — no deduplication by file_id — matching the
    behaviour of OpenAI's native file_search which surfaces every relevant
    chunk even when multiple chunks originate from the same document.
    """
    formatted: List[Dict[str, Any]] = []
    for result in results:
        file_id = _get_field(result, "file_id") or ""
        content_items = _get_field(result, "content") or []
        text_chunks = [
            c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
            for c in content_items
        ]
        text = " ".join(t for t in text_chunks if t)
        formatted.append(
            {
                "file_id": file_id,
                "filename": _get_field(result, "filename") or "",
                "score": _get_field(result, "score"),
                "text": text,
                "attributes": _get_field(result, "attributes") or {},
            }
        )
    return formatted


def _build_file_search_call_output(
    call_id: str,
    queries: List[str],
    results: Optional[List[VectorStoreSearchResult]] = None,
    include_search_results: bool = False,
) -> Dict[str, Any]:
    """Build the file_search_call output item (mirrors OpenAI's format).

    Args:
        call_id: Unique ID for this file_search call.
        queries: List of search queries used.
        results: The raw search results (used when include_search_results=True).
        include_search_results: Populate search_results when the caller passed
            ``include=["file_search_call.results"]``.
    """
    search_results = None
    if include_search_results and results:
        search_results = _build_search_results_for_include(results)
    return {
        "type": "file_search_call",
        "id": call_id,
        "status": "completed",
        "queries": queries,
        "search_results": search_results,
    }


def _build_file_citation_annotations(
    results: List[VectorStoreSearchResult],
    text: str,
) -> List[Dict[str, Any]]:
    """
    Build file_citation annotations for the text.
    Each result with a file_id gets a citation at the end of the text.
    """
    annotations: List[Dict[str, Any]] = []
    index = len(text)  # cite at end of text block
    seen_file_ids: set = set()

    for result in results:
        file_id = _get_field(result, "file_id")
        filename = _get_field(result, "filename")
        if not file_id or file_id in seen_file_ids:
            continue
        seen_file_ids.add(file_id)
        annotations.append(
            {
                "type": "file_citation",
                "index": index,
                "file_id": file_id,
                "filename": filename or "",
            }
        )

    return annotations


def _build_message_output(
    response_text: str,
    results: List[VectorStoreSearchResult],
) -> Dict[str, Any]:
    """Build the message output item with optional file_citation annotations."""
    annotations = _build_file_citation_annotations(results, response_text)
    return {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": response_text,
                "annotations": annotations,
            }
        ],
    }


def _extract_text_from_responses_output(response: ResponsesAPIResponse) -> str:
    """Pull the assistant's text from the provider's response."""
    for item in response.output:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type == "message":
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", [])
            for block in (content or []):
                block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                if block_type == "output_text":
                    raw = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
                    return str(raw) if raw is not None else ""
    return ""


def _synthesize_responses_api_response(
    original_response: ResponsesAPIResponse,
    file_search_call_output: Dict[str, Any],
    message_output: Dict[str, Any],
    first_response: Optional[ResponsesAPIResponse] = None,
) -> ResponsesAPIResponse:
    """
    Return a new ResponsesAPIResponse with:
      output[0] = file_search_call item
      output[1] = message item (with citations)

    When first_response is provided, its response_cost is accumulated into the
    synthesized _hidden_params so that billing callbacks see the total cost of
    both provider calls that the emulated flow makes.
    """
    synthesized = ResponsesAPIResponse(
        id=getattr(original_response, "id", f"resp_{uuid.uuid4().hex}"),
        object="response",
        created_at=getattr(original_response, "created_at", int(time.time())),
        status="completed",
        model=getattr(original_response, "model", ""),
        output=[file_search_call_output, message_output],
        usage=getattr(original_response, "usage", None),
        error=None,
    )
    if hasattr(original_response, "_hidden_params"):
        hidden = dict(getattr(original_response, "_hidden_params") or {})
        if first_response is not None and hasattr(first_response, "_hidden_params"):
            first_hidden = getattr(first_response, "_hidden_params") or {}
            first_cost = first_hidden.get("response_cost") if isinstance(first_hidden, dict) else getattr(first_hidden, "response_cost", None)
            if first_cost is not None:
                current_cost = hidden.get("response_cost") if isinstance(hidden, dict) else 0
                hidden["response_cost"] = (current_cost or 0) + first_cost
        synthesized._hidden_params = hidden
    return synthesized


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def _call_aresponses(input, model, tools, **kwargs):  # pragma: no cover – thin wrapper for patching in tests
    from litellm.responses.main import aresponses
    return await aresponses(input=input, model=model, tools=tools, **kwargs)


async def aresponses_with_emulated_file_search(
    input: Any,
    model: str,
    tools: Optional[Iterable[ToolParam]] = None,
    # Pass-through params — forwarded as-is to the underlying aresponses call
    **kwargs: Any,
) -> ResponsesAPIResponse:
    """
    Emulated file_search for providers that don't support it natively.

    Replaces file_search tools with a function tool, intercepts the tool call,
    runs vector search, and synthesizes an OpenAI-format response.
    """
    # Determine whether caller wants search_results populated in the output.
    _include: List[str] = list(kwargs.get("include") or [])
    _include_search_results = "file_search_call.results" in _include

    # Disable streaming for emulated file_search (not yet supported)
    _original_stream = kwargs.get("stream")
    if _original_stream:
        verbose_logger.debug(
            "Streaming is not yet supported for emulated file_search. "
            "Disabling stream for this request."
        )
        kwargs = {**kwargs, "stream": False}

    # 1. Replace file_search tools with function tool
    transformed_tools, all_vs_ids = _replace_file_search_tools(tools)

    # 2. First provider call — provider will call the file_search function
    first_response: ResponsesAPIResponse = cast(
        ResponsesAPIResponse,
        await _call_aresponses(
            input=input,
            model=model,
            tools=transformed_tools or None,
            **kwargs,
        ),
    )

    # 3. Look for a file_search function_call in the output
    file_search_calls = [
        item
        for item in first_response.output
        if (
            isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("name") == FILE_SEARCH_FUNCTION_NAME
        )
        or (
            hasattr(item, "type")
            and getattr(item, "type") == "function_call"
            and getattr(item, "name", None) == FILE_SEARCH_FUNCTION_NAME
        )
    ]

    if not file_search_calls:
        # Provider answered without calling the tool (e.g. it had enough context).
        # Return as-is wrapped in OpenAI format.
        call_id = f"fs_{uuid.uuid4().hex[:24]}"
        response_text = _extract_text_from_responses_output(first_response)
        return _synthesize_responses_api_response(
            original_response=first_response,
            file_search_call_output=_build_file_search_call_output(
                call_id=call_id,
                queries=[str(input)],
                results=None,
                include_search_results=False,
            ),
            message_output=_build_message_output(response_text, []),
        )

    # 4. Execute each file_search tool call
    tool_results: List[Dict[str, Any]] = []
    all_queries: List[str] = []
    all_results: List[VectorStoreSearchResult] = []
    file_search_call_id = f"fs_{uuid.uuid4().hex[:24]}"

    for tool_call in file_search_calls:
        if isinstance(tool_call, dict):
            call_id = tool_call.get("call_id") or tool_call.get("id") or file_search_call_id
            raw_args = tool_call.get("arguments") or "{}"
        else:
            call_id = getattr(tool_call, "call_id", None) or getattr(tool_call, "id", file_search_call_id)
            raw_args = getattr(tool_call, "arguments", "{}") or "{}"

        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {}

        # Extract queries array (OpenAI-style multi-query support)
        queries_from_call = args.get("queries")
        if not queries_from_call:
            # Fallback: check for single "query" field (backward compat)
            single_query = args.get("query")
            queries_from_call = [single_query] if single_query else [str(input)]
        elif not isinstance(queries_from_call, list):
            queries_from_call = [str(queries_from_call)]

        vs_id_arg = args.get("vector_store_id")
        vs_ids_for_call = [vs_id_arg] if vs_id_arg else all_vs_ids

        queries, results = await _run_vector_searches(
            queries=queries_from_call,
            vector_store_ids=vs_ids_for_call,
        )
        all_queries.extend(queries)
        all_results.extend(results)

        tool_results.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": _format_search_results_as_tool_output(results),
            }
        )

    # 5. Build follow-up input: original messages + ALL first-response output items + tool results
    # Including all output items (text blocks, reasoning, non-file-search calls) ensures providers
    # like Anthropic that emit text before the tool call have complete conversation context.
    # Serialize Pydantic model instances to plain dicts so the transformation layer can call .get().
    original_input_items = list(input) if isinstance(input, (list, tuple)) else [{"role": "user", "content": str(input)}]
    first_response_output_items: List[Any] = []
    for _item in first_response.output:
        if isinstance(_item, dict):
            first_response_output_items.append(_item)
        elif hasattr(_item, "model_dump"):
            first_response_output_items.append(_item.model_dump(exclude_none=True))  # type: ignore[union-attr]
        else:
            first_response_output_items.append(_item)

    follow_up_input = (
        original_input_items
        + first_response_output_items
        + tool_results
    )

    # 6. Follow-up call — provider writes the final answer given search results
    final_response: ResponsesAPIResponse = cast(
        ResponsesAPIResponse,
        await _call_aresponses(
            input=follow_up_input,
            model=model,
            tools=None,  # no tools needed for the answer step
            **kwargs,
        ),
    )

    # 7. Synthesize OpenAI-format output
    response_text = _extract_text_from_responses_output(final_response)

    return _synthesize_responses_api_response(
        original_response=final_response,
        file_search_call_output=_build_file_search_call_output(
            call_id=file_search_call_id,
            queries=all_queries or [str(input)],
            results=all_results,
            include_search_results=_include_search_results,
        ),
        message_output=_build_message_output(response_text, all_results),
        first_response=first_response,
    )
