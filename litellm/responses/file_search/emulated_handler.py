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

import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypeAlias, TypedDict, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm._internal_context import is_internal_call
from litellm._logging import verbose_logger
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponseOutputItem,
    ResponsesAPIResponse,
    ToolParam,
)
from litellm.types.vector_stores import (
    VectorStoreResultContent,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

FILE_SEARCH_FUNCTION_NAME: Final = "litellm_file_search"

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class SearchResultContentLike(Protocol):
    """Attribute-based counterpart of ``VectorStoreResultContent``."""

    @property
    def text(self) -> str | None: ...


class SearchResultLike(Protocol):
    """Attribute-based counterpart of ``VectorStoreSearchResult``."""

    @property
    def score(self) -> float | None: ...

    @property
    def file_id(self) -> str | None: ...

    @property
    def filename(self) -> str | None: ...

    @property
    def content(self) -> Sequence[VectorStoreResultContent | SearchResultContentLike] | None: ...

    @property
    def attributes(self) -> Mapping[str, object] | None: ...


class SearchResponseLike(Protocol):
    """Attribute-based counterpart of ``VectorStoreSearchResponse``."""

    @property
    def data(self) -> Sequence["SearchResult"] | None: ...


SearchResult: TypeAlias = VectorStoreSearchResult | SearchResultLike


class FunctionToolQueriesSchema(TypedDict):
    type: Literal["array"]
    items: dict[str, str]
    description: str


class FunctionToolVectorStoreIdSchema(TypedDict):
    type: Literal["string"]
    description: str
    enum: list[str]


class FunctionToolProperties(TypedDict):
    queries: FunctionToolQueriesSchema
    vector_store_id: FunctionToolVectorStoreIdSchema


class FunctionToolParameters(TypedDict):
    type: Literal["object"]
    properties: FunctionToolProperties
    required: list[str]


class EmulatedFileSearchTool(TypedDict):
    type: Literal["function"]
    name: str
    description: str
    parameters: FunctionToolParameters


class FileCitationAnnotation(TypedDict):
    type: Literal["file_citation"]
    index: int
    file_id: str
    filename: str


class OutputTextContent(TypedDict):
    type: Literal["output_text"]
    text: str
    annotations: list[FileCitationAnnotation]


class MessageOutput(TypedDict):
    type: Literal["message"]
    role: Literal["assistant"]
    content: list[OutputTextContent]


class FileSearchResultEntry(TypedDict):
    file_id: str
    filename: str
    score: float | None
    text: str
    attributes: Mapping[str, object]


class FileSearchCallOutput(TypedDict):
    type: Literal["file_search_call"]
    id: str
    status: Literal["completed"]
    queries: list[str]
    search_results: list[FileSearchResultEntry] | None


class FunctionCallOutput(TypedDict):
    type: Literal["function_call_output"]
    call_id: str
    output: str


class FileSearchToolSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["file_search"]
    vector_store_ids: tuple[str, ...] | None = None


class FileSearchFunctionCall(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    type: Literal["function_call"]
    name: str | None = None
    call_id: str | None = None
    id: str | None = None
    arguments: str | Mapping[str, object] | None = None


class FileSearchArguments(BaseModel):
    model_config = ConfigDict(extra="ignore")

    queries: tuple[str, ...] | str | None = None
    query: str | None = None
    vector_store_id: str | None = None


class MessageContentBlock(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    type: str | None = None
    text: str | None = None


class MessageOutputItem(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    type: Literal["message"]
    content: tuple[MessageContentBlock, ...] | None = None


class IncludeOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    include: tuple[str, ...] | None = None


def _validate_or_none(model: type[_ModelT], value: object) -> _ModelT | None:
    try:
        return model.model_validate(value)
    except ValidationError:
        return None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _as_file_search_tool(tool: object) -> FileSearchToolSpec | None:
    return _validate_or_none(FileSearchToolSpec, tool)


def should_use_emulated_file_search(
    tools: Iterable[ToolParam] | None,
    provider_config: BaseResponsesAPIConfig | None,
) -> bool:
    """Return True when there is a file_search tool and the provider can't handle it natively."""
    if not tools:
        return False
    has_fs: Final = any(_as_file_search_tool(tool) is not None for tool in tools)
    if not has_fs:
        return False
    return provider_config is None or not provider_config.supports_native_file_search()


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


def _build_function_tool(vector_store_ids: list[str]) -> EmulatedFileSearchTool:
    """
    Create a Responses API function-tool definition that describes file search.
    The function accepts one or more natural-language queries (like OpenAI's native
    file_search); LiteLLM runs the actual vector search against the configured
    vector stores.

    Note: Uses Responses API format (name/description/parameters at top level),
    NOT Chat Completion format (nested under "function"), so that the
    LiteLLMCompletionResponsesConfig transformation picks up name and description.
    """
    return EmulatedFileSearchTool(
        type="function",
        name=FILE_SEARCH_FUNCTION_NAME,
        description=(
            "Search the knowledge base for information relevant to the query. "
            "Use this whenever you need to look up specific facts, documents, "
            "or content from the vector store. You can provide multiple queries "
            "to search for different aspects of the information."
        ),
        parameters=FunctionToolParameters(
            type="object",
            properties=FunctionToolProperties(
                queries=FunctionToolQueriesSchema(
                    type="array",
                    items={"type": "string"},
                    description=(
                        "One or more search queries to look up in the vector store. "
                        "Multiple queries help find comprehensive information from "
                        "different angles."
                    ),
                ),
                vector_store_id=FunctionToolVectorStoreIdSchema(
                    type="string",
                    description="ID of the vector store to search.",
                    enum=vector_store_ids,
                ),
            ),
            required=["queries"],
        ),
    )


def _replace_file_search_tools(
    tools: Iterable[ToolParam] | None,
) -> tuple[list[ToolParam | EmulatedFileSearchTool], list[str]]:
    """
    Replace all file_search tools with a single function tool.

    Returns:
        (new_tools_list, all_vector_store_ids)
    """
    parsed: Final = tuple((tool, _as_file_search_tool(tool)) for tool in tools or ())
    non_file_search: Final[list[ToolParam | EmulatedFileSearchTool]] = [
        tool for tool, file_search in parsed if file_search is None
    ]
    unique_ids: Final[list[str]] = list(
        dict.fromkeys(
            vector_store_id
            for _, file_search in parsed
            if file_search is not None
            for vector_store_id in file_search.vector_store_ids or ()
        )
    )
    if not unique_ids:
        return non_file_search, unique_ids
    return [*non_file_search, _build_function_tool(unique_ids)], unique_ids


# ---------------------------------------------------------------------------
# Search execution
# ---------------------------------------------------------------------------


def _results_of(response: VectorStoreSearchResponse | SearchResponseLike) -> tuple[SearchResult, ...]:
    results_data: Final = response.get("data") if isinstance(response, dict) else response.data
    return tuple(results_data or ())


async def _search_one_vector_store(vector_store_id: str, query: str) -> tuple[SearchResult, ...]:
    """Run a single ``asearch`` call, returning no results when the search fails."""
    import litellm.vector_stores.main as vs_main

    try:
        return _results_of(await vs_main.asearch(vector_store_id=vector_store_id, query=query))
    except Exception as exc:
        verbose_logger.warning(
            "file_search emulated: search failed for query='%s', vector_store_id='%s': %s",
            query,
            vector_store_id,
            exc,
        )
        return ()


async def _run_vector_searches(
    queries: list[str],
    vector_store_ids: list[str],
) -> tuple[list[str], list[SearchResult]]:
    """
    Run `asearch` against all vector stores for all queries and collect results.

    Args:
        queries: List of search queries to execute (like OpenAI's multi-query approach)
        vector_store_ids: Vector store IDs to search

    Returns:
        (queries_list, combined_results)
    """
    all_results: Final[list[SearchResult]] = [
        result
        for query in queries
        for vector_store_id in vector_store_ids
        for result in await _search_one_vector_store(vector_store_id=vector_store_id, query=query)
    ]
    return queries, all_results


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchResultView:
    """Normalized view over a dict- or attribute-shaped vector store search result."""

    score: float | None
    file_id: str
    filename: str
    text: str
    attributes: Mapping[str, object]


def _content_text(item: VectorStoreResultContent | SearchResultContentLike) -> str:
    text: Final = item.get("text") if isinstance(item, dict) else item.text
    return text or ""


def _view_of(result: SearchResult) -> SearchResultView:
    if isinstance(result, dict):
        score, file_id, filename = result.get("score"), result.get("file_id"), result.get("filename")
        content, attributes = result.get("content"), result.get("attributes")
    else:
        score, file_id, filename = result.score, result.file_id, result.filename
        content, attributes = result.content, result.attributes
    text: Final = " ".join(chunk for chunk in (_content_text(item) for item in content or ()) if chunk)
    return SearchResultView(
        score=score,
        file_id=file_id or "",
        filename=filename or "",
        text=text,
        attributes=attributes or {},
    )


def _result_header(index: int, view: SearchResultView) -> str:
    segments: Final = (
        f"Result {index}",
        view.filename or None,
        f"file_id={view.file_id}" if view.file_id else None,
        f"score={view.score:.3f}" if view.score is not None else None,
    )
    return f"[{' | '.join(segment for segment in segments if segment)}]"


def _format_search_results_as_tool_output(
    results: Sequence[SearchResult],
) -> str:
    """Serialize search results into a string to pass back as the tool's output."""
    if not results:
        return "No results found in the vector store."

    views: Final = tuple(_view_of(result) for result in results)
    return "\n\n".join(f"{_result_header(index, view)}\n{view.text}" for index, view in enumerate(views, 1))


def _build_search_results_for_include(
    results: Sequence[SearchResult],
) -> list[FileSearchResultEntry]:
    """
    Convert VectorStoreSearchResult objects to the format expected in
    file_search_call.search_results (mirrors OpenAI's include= format).

    All chunks are returned, with no deduplication by file_id, matching the
    behaviour of OpenAI's native file_search which surfaces every relevant
    chunk even when multiple chunks originate from the same document.
    """
    return [
        FileSearchResultEntry(
            file_id=view.file_id,
            filename=view.filename,
            score=view.score,
            text=view.text,
            attributes=view.attributes,
        )
        for view in (_view_of(result) for result in results)
    ]


def _build_file_search_call_output(
    call_id: str,
    queries: list[str],
    results: Sequence[SearchResult] | None = None,
    include_search_results: bool = False,
) -> FileSearchCallOutput:
    """Build the file_search_call output item (mirrors OpenAI's format).

    Args:
        call_id: Unique ID for this file_search call.
        queries: List of search queries used.
        results: The raw search results (used when include_search_results=True).
        include_search_results: Populate search_results when the caller passed
            ``include=["file_search_call.results"]``.
    """
    search_results: Final = _build_search_results_for_include(results) if include_search_results and results else None
    return FileSearchCallOutput(
        type="file_search_call",
        id=call_id,
        status="completed",
        queries=queries,
        search_results=search_results,
    )


def _build_file_citation_annotations(
    results: Sequence[SearchResult],
    text: str,
) -> list[FileCitationAnnotation]:
    """
    Build file_citation annotations for the text.
    Each result with a file_id gets a citation at the end of the text.
    """
    views: Final = tuple(_view_of(result) for result in results)
    ordered_file_ids: Final = tuple(dict.fromkeys(view.file_id for view in views if view.file_id))
    filename_by_file_id: Final = {view.file_id: view.filename for view in reversed(views) if view.file_id}
    return [
        FileCitationAnnotation(
            type="file_citation",
            index=len(text),
            file_id=file_id,
            filename=filename_by_file_id.get(file_id, ""),
        )
        for file_id in ordered_file_ids
    ]


def _build_message_output(
    response_text: str,
    results: Sequence[SearchResult],
) -> MessageOutput:
    """Build the message output item with optional file_citation annotations."""
    return MessageOutput(
        type="message",
        role="assistant",
        content=[
            OutputTextContent(
                type="output_text",
                text=response_text,
                annotations=_build_file_citation_annotations(results, response_text),
            )
        ],
    )


def _output_items_of(response: ResponsesAPIResponse) -> tuple[object, ...]:
    return tuple(response.output)


def _extract_text_from_responses_output(response: ResponsesAPIResponse) -> str:
    """Pull the assistant's text from the provider's response."""
    messages: Final = tuple(
        message
        for message in (_validate_or_none(MessageOutputItem, item) for item in _output_items_of(response))
        if message is not None
    )
    texts: Final = tuple(
        block.text or "" for message in messages for block in message.content or () if block.type == "output_text"
    )
    return texts[0] if texts else ""


def _response_cost(hidden_params: Mapping[str, object]) -> float | None:
    cost: Final = hidden_params.get("response_cost")
    return float(cost) if isinstance(cost, (int, float)) else None


def _hidden_params_of(response: ResponsesAPIResponse) -> Mapping[str, object]:
    raw: Final[Mapping[str, object] | None] = getattr(response, "_hidden_params", None)
    return raw or {}


def _synthesize_responses_api_response(
    original_response: ResponsesAPIResponse,
    file_search_call_output: FileSearchCallOutput,
    message_output: MessageOutput,
    first_response: ResponsesAPIResponse | None = None,
) -> ResponsesAPIResponse:
    """
    Return a new ResponsesAPIResponse with:
      output[0] = file_search_call item
      output[1] = message item (with citations)

    When first_response is provided, its response_cost is accumulated into the
    synthesized _hidden_params so that billing callbacks see the total cost of
    both provider calls that the emulated flow makes.
    """
    synthesized_output: Final[list[ResponseOutputItem | dict[str, object]]] = [
        dict(file_search_call_output),
        dict(message_output),
    ]
    synthesized: Final = ResponsesAPIResponse(
        id=original_response.id,
        object="response",
        created_at=original_response.created_at,
        status="completed",
        model=original_response.model,
        output=synthesized_output,
        usage=original_response.usage,
        error=None,
    )
    hidden: Final = _hidden_params_of(original_response)
    first_cost: Final = _response_cost(_hidden_params_of(first_response)) if first_response is not None else None
    synthesized._hidden_params = (
        {**hidden} if first_cost is None else {**hidden, "response_cost": (_response_cost(hidden) or 0) + first_cost}
    )
    return synthesized


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


# any-ok: thin, test-patchable seam forwarding provider pass-through params to aresponses
async def _call_aresponses(input, model: str, tools, **kwargs) -> ResponsesAPIResponse:
    from litellm.responses.main import aresponses

    response: Final = await aresponses(input=input, model=model, tools=tools, **kwargs)
    if isinstance(response, ResponsesAPIResponse):
        return response
    raise ValueError("emulated file_search does not support streaming responses")


def _prepare_emulated_file_search_call(
    kwargs: Mapping[str, object],
) -> tuple[bool, dict[str, object]]:
    include_options: Final = _validate_or_none(IncludeOptions, kwargs)
    include_search_results: Final = "file_search_call.results" in ((include_options and include_options.include) or ())

    if not kwargs.get("stream"):
        return include_search_results, {**kwargs}

    verbose_logger.debug("Streaming is not yet supported for emulated file_search. Disabling stream for this request.")
    return include_search_results, {**kwargs, "stream": False}


def _resolve_queries_from_args(args: FileSearchArguments, input: str | ResponseInputParam) -> list[str]:
    """Pull the queries list out of parsed tool-call arguments, with backward-compat fallbacks."""
    if not args.queries:
        return [args.query] if args.query else [str(input)]
    if isinstance(args.queries, str):
        return [args.queries]
    return list(args.queries)


def _parse_file_search_arguments(raw_arguments: str | Mapping[str, object] | None) -> FileSearchArguments:
    if isinstance(raw_arguments, str):
        try:
            return FileSearchArguments.model_validate_json(raw_arguments)
        except ValidationError:
            return FileSearchArguments()
    return _validate_or_none(FileSearchArguments, raw_arguments) or FileSearchArguments()


async def _execute_file_search_tool_calls(
    file_search_calls: Sequence[FileSearchFunctionCall],
    all_vs_ids: list[str],
    input: str | ResponseInputParam,
    file_search_call_id: str,
) -> tuple[list[FunctionCallOutput], list[str], list[SearchResult]]:
    """Run the vector search for each file_search tool_call and collect results."""
    tool_results: Final[list[FunctionCallOutput]] = []
    all_queries: Final[list[str]] = []
    all_results: Final[list[SearchResult]] = []

    for tool_call in file_search_calls:
        call_id = str(tool_call.call_id or tool_call.id or file_search_call_id)
        args = _parse_file_search_arguments(tool_call.arguments)
        queries, results = await _run_vector_searches(
            queries=_resolve_queries_from_args(args, input),
            vector_store_ids=[args.vector_store_id] if args.vector_store_id else all_vs_ids,
        )
        all_queries.extend(queries)
        all_results.extend(results)

        tool_results.append(
            FunctionCallOutput(
                type="function_call_output",
                call_id=call_id,
                output=_format_search_results_as_tool_output(results),
            )
        )

    return tool_results, all_queries, all_results


def _as_plain_input_item(item: object) -> object:
    return item.model_dump(exclude_none=True) if isinstance(item, BaseModel) else item


def _build_follow_up_input(
    input: str | ResponseInputParam,
    first_response: ResponsesAPIResponse,
    tool_results: Sequence[FunctionCallOutput],
) -> list[object]:
    """Assemble the follow-up call input: original messages + first-response output + tool results.

    Including all output items (text blocks, reasoning, non-file-search calls) ensures providers
    like Anthropic that emit text before the tool call have complete conversation context.
    Serializes Pydantic model instances to plain dicts so the transformation layer can call .get().
    """
    original_input_items: Final[list[object]] = (
        list(input) if isinstance(input, (list, tuple)) else [{"role": "user", "content": str(input)}]
    )
    return [
        *original_input_items,
        *(_as_plain_input_item(item) for item in _output_items_of(first_response)),
        *tool_results,
    ]


def _file_search_calls_in(response: ResponsesAPIResponse) -> tuple[FileSearchFunctionCall, ...]:
    return tuple(
        call
        for call in (_validate_or_none(FileSearchFunctionCall, item) for item in _output_items_of(response))
        if call is not None and call.name == FILE_SEARCH_FUNCTION_NAME
    )


async def aresponses_with_emulated_file_search(
    input: str | ResponseInputParam,
    model: str,
    tools: Iterable[ToolParam] | None = None,
    # Pass-through params — forwarded as-is to the underlying aresponses call
    **kwargs: object,
) -> ResponsesAPIResponse:
    """
    Emulated file_search for providers that don't support it natively.

    Replaces file_search tools with a function tool, intercepts the tool call,
    runs vector search, and synthesizes an OpenAI-format response.
    """
    # Determine whether caller wants search_results populated in the output.
    include_search_results, forwarded_kwargs = _prepare_emulated_file_search_call(kwargs=kwargs)

    # 1. Replace file_search tools with function tool
    transformed_tools, all_vs_ids = _replace_file_search_tools(tools)

    # 2. First provider call — provider will call the file_search function.
    # Mark as an internal sub-call so wrapper_async skips billing callbacks;
    # the parent litellm_logging_obj (propagated via kwargs) fires once at the end.
    prev_internal: Final = is_internal_call.get()
    is_internal_call.set(True)
    try:
        first_response: Final = await _call_aresponses(
            input=input,
            model=model,
            tools=transformed_tools or None,
            **forwarded_kwargs,
        )
    finally:
        is_internal_call.set(prev_internal)

    # 3. Look for a file_search function_call in the output
    file_search_calls: Final = _file_search_calls_in(first_response)

    if not file_search_calls:
        # Provider answered without calling the tool (e.g. it had enough context).
        # Return as-is wrapped in OpenAI format.
        call_id: Final = f"fs_{uuid.uuid4().hex[:24]}"
        return _synthesize_responses_api_response(
            original_response=first_response,
            file_search_call_output=_build_file_search_call_output(
                call_id=call_id,
                queries=[str(input)],
                results=None,
                include_search_results=False,
            ),
            message_output=_build_message_output(_extract_text_from_responses_output(first_response), []),
        )

    # 4. Execute each file_search tool call
    file_search_call_id: Final = f"fs_{uuid.uuid4().hex[:24]}"
    tool_results, all_queries, all_results = await _execute_file_search_tool_calls(
        file_search_calls=file_search_calls,
        all_vs_ids=all_vs_ids,
        input=input,
        file_search_call_id=file_search_call_id,
    )

    # 5. Build follow-up input: original messages + ALL first-response output items + tool results
    follow_up_input: Final = _build_follow_up_input(
        input=input,
        first_response=first_response,
        tool_results=tool_results,
    )

    # 6. Follow-up call — provider writes the final answer given search results.
    # Also an internal sub-call; billing is suppressed so the outer call fires once.
    is_internal_call.set(True)
    try:
        final_response: Final = await _call_aresponses(
            input=follow_up_input,
            model=model,
            tools=None,  # no tools needed for the answer step
            **forwarded_kwargs,
        )
    finally:
        is_internal_call.set(prev_internal)

    # 7. Synthesize OpenAI-format output
    return _synthesize_responses_api_response(
        original_response=final_response,
        file_search_call_output=_build_file_search_call_output(
            call_id=file_search_call_id,
            queries=all_queries or [str(input)],
            results=all_results,
            include_search_results=include_search_results,
        ),
        message_output=_build_message_output(_extract_text_from_responses_output(final_response), all_results),
        first_response=first_response,
    )
