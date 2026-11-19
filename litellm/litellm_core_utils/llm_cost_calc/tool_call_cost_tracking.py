"""
Which built-in tools a call used, read off its response and request, for Rust to price.

Rust prices every tool fee (``NativeModelCostMap.price_tools`` and the token receipt). This module reads the counts
off the response, looks up the rows main's calculators read and keeps the public calculator signatures as thin
wrappers over one Rust call each.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, cast

from pydantic import ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.constants import (
    AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS,
    AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS,
    AZURE_FILE_SEARCH_COST_PER_GB_PER_DAY,
    AZURE_VECTOR_STORE_COST_PER_GB_PER_DAY,
    OPENAI_FILE_SEARCH_COST_PER_1K_CALLS,
)
from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import (
    CodeInterpreterTool,
    ComputerUseTool,
    ImageGenerationTool,
    ImageItem,
    ImageTokens,
    MapsGroundingTool,
    OneCodeInterpreter,
    OneComputerUse,
    OneFileSearch,
    OneToolOnly,
    OneVectorStore,
    OneWebSearch,
    SearchContextSize,
    SearchCounts,
    ToolsOfCall,
    ToolUse,
    VectorStoreTool,
    WebSearchDispatch,
    WebSearchPerCall,
    WebSearchTool,
    price_tools,
)
from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import (
    FileSearchTool as FileSearchUse,
)
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    get_web_search_requests_from_usage,
)
from litellm.types.llms.openai import (
    FileSearchTool,
    ResponsesAPIResponse,
    ResponsesToolUsage,
    WebSearchOptions,
)
from litellm.types.utils import (
    ChatCompletionAnnotation,
    Message,
    ModelInfo,
    ModelResponse,
    PromptTokensDetailsWrapper,
    ServerToolUse,
    StandardBuiltInToolsParams,
    Usage,
)

_NO_ROWS: Final[tuple[Mapping[str, object], ...]] = ()


def _output_item_field(output_item: object, field: str) -> object:
    if isinstance(output_item, dict):
        fields: Final[Mapping[str, object]] = cast(Mapping[str, object], output_item)  # cast-ok: narrowed by isinstance
        return fields.get(field)
    return getattr(output_item, field, None)


def _output_item_type(output_item: object) -> str | None:
    item_type: Final = _output_item_field(output_item, "type")
    return item_type if isinstance(item_type, str) else None


def _responses_tool_usage(response_object: ResponsesAPIResponse) -> ResponsesToolUsage | None:
    top_level: Final = getattr(response_object, "tool_usage", None)
    raw: Final = (
        top_level if top_level is not None else getattr(getattr(response_object, "usage", None), "tool_usage", None)
    )
    if raw is None:
        return None
    try:
        return ResponsesToolUsage.model_validate(raw)
    except ValidationError:
        return None


def _reported_web_search_requests(response_object: ResponsesAPIResponse) -> int | None:
    tool_usage: Final = _responses_tool_usage(response_object)
    if tool_usage is None:
        return None
    return None if tool_usage.web_search is None else tool_usage.web_search.num_requests


def _usage_reports_server_side_web_search_calls(usage: Usage) -> bool:
    details: Final = getattr(usage, "server_side_tool_usage_details", None)
    if not isinstance(details, Mapping):
        return False
    calls: Final = details.get("web_search_calls")
    return isinstance(calls, int) and calls > 0


def _requests(value: object) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return max(value, 0)


def _search_context(size: object) -> SearchContextSize | None:
    if size == "low":
        return "low"
    if size == "medium":
        return "medium"
    return "high" if size == "high" else None


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _rows_at(
    rows: Sequence[Mapping[str, object] | None],
) -> tuple[tuple[int | None, ...], tuple[Mapping[str, object], ...]]:
    """Where each of ``rows`` lands among the rows passed to Rust, ``None`` for a row that does not exist."""
    present: Final = tuple(row for row in rows if row is not None)
    positions: Final = tuple(
        None if row is None else sum(1 for earlier in rows[:index] if earlier is not None)
        for index, row in enumerate(rows)
    )
    return positions, present


def search_counts(usage: Usage) -> SearchCounts:
    """The web search counts ``usage`` reports, each provider's calculator reading its own."""
    from litellm.llms.xai.cost_calculator import stated_cost

    details: Final[object] = getattr(usage, "prompt_tokens_details", None)
    server_tool_use: Final[object] = getattr(usage, "server_tool_use", None)
    side_details: Final = getattr(usage, "server_side_tool_usage_details", None)
    side_calls: Final[object] = side_details.get("web_search_calls") if isinstance(side_details, Mapping) else None
    return SearchCounts(
        prompt_requests=_requests(getattr(details, "web_search_requests", None))
        if isinstance(details, PromptTokensDetailsWrapper)
        else None,
        server_requests=_requests(get_web_search_requests_from_usage(usage)),
        browser_opens=_requests(getattr(server_tool_use, "browser_open_requests", None)),
        search_calls=_search_calls(side_calls),
        stated_cost=stated_cost(usage) is not None,
    )


def _search_calls(value: object) -> int:
    if not value or not isinstance(value, (int, float, str)):
        return 0
    try:
        return max(int(value), 0)
    except ValueError:
        return 0


def _served_provider(provider: str) -> str | None:
    """The provider the served model's row is looked up under, for the calculators that borrow its search prices."""
    if provider in ("anthropic", "azure_ai"):
        return provider
    return "vertex_ai" if provider.startswith("vertex_ai") else None


def _served_row(
    model_info: ModelInfo,
    provider: str,
    response_model: str | None,
    lookup_model_info: Callable[[str, str | None], ModelInfo | None] | None,
) -> ModelInfo | None:
    served_provider: Final = _served_provider(provider)
    if served_provider is None or response_model is None or lookup_model_info is None:
        return None
    if model_info.get("search_context_cost_per_query") is not None:
        return None
    return lookup_model_info(response_model, served_provider)


def web_search_dispatch(
    provider: str,
    usage: Usage | None,
    model_info: ModelInfo | None,
    response_model: str | None = None,
    lookup_model_info: Callable[[str, str | None], ModelInfo | None] | None = None,
) -> tuple[WebSearchDispatch, tuple[Mapping[str, object], ...]]:
    """The facts ``get_cost_for_web_search_request`` dispatches with and the rows they name."""
    served: Final = None if model_info is None else _served_row(model_info, provider, response_model, lookup_model_info)
    (row, served_row), rows = _rows_at((model_info, served))
    key: Final[object] = None if model_info is None else model_info.get("key")
    return (
        WebSearchDispatch(
            provider=provider,
            row=row,
            served_row=served_row,
            model_key=key if isinstance(key, str) else "",
            response_model=response_model,
            counts=search_counts(usage)
            if usage is not None
            else SearchCounts(
                prompt_requests=None, server_requests=None, browser_opens=None, search_calls=0, stated_cost=False
            ),
        ),
        rows,
    )


def price_web_search_dispatch(
    provider: str,
    usage: Usage | None,
    model_info: ModelInfo | None,
    response_model: str | None = None,
    lookup_model_info: Callable[[str, str | None], ModelInfo | None] | None = None,
) -> float | None:
    """``provider``'s web search fee on ``model_info``, ``None`` for a provider without a calculator of its own."""
    dispatch, rows = web_search_dispatch(provider, usage, model_info, response_model, lookup_model_info)
    return price_tools(OneToolOnly(one=OneWebSearch(tool="web_search", dispatch=dispatch, per_call=None)), rows)


def _container_row(provider: str | None) -> Mapping[str, object] | None:
    """The ``<provider>/container`` row a code interpreter session falls back to."""
    if not provider:
        return None
    try:
        model_info: Final = litellm.get_model_info(model=f"{provider}/container", custom_llm_provider=provider)
    except Exception:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models
        return None
    key: Final[object] = model_info.get("key")
    if not isinstance(key, str):
        return None
    row: Final[object] = litellm.model_cost.get(key)  # pyright: ignore[reportUnknownMemberType]  # model cost map is dynamically loaded
    return cast(Mapping[str, object], row) if isinstance(row, dict) else None  # cast-ok: narrowed by isinstance


class StandardBuiltInToolCostTracking:
    """
    Reads the built-in tools a call used, such as web search, and has Rust price them.
    """

    @staticmethod
    def get_cost_for_built_in_tools(
        model: str,
        response_object: object,
        usage: Usage | None = None,
        custom_llm_provider: str | None = None,
        standard_built_in_tools_params: StandardBuiltInToolsParams | None = None,
    ) -> float:
        """
        Get the cost of using built-in tools: web search, file search, Maps grounding, the Responses image
        generation tool and Azure's vector store, computer use and code interpreter.
        """
        used, rows = StandardBuiltInToolCostTracking.built_in_tool_use(
            model=model,
            response_object=response_object,
            usage=usage,
            custom_llm_provider=custom_llm_provider,
            standard_built_in_tools_params=standard_built_in_tools_params,
        )
        return price_tools(ToolsOfCall(call=used), rows) or 0.0

    @staticmethod
    def built_in_tool_use(
        model: str,
        response_object: object,
        usage: Usage | None = None,
        custom_llm_provider: str | None = None,
        standard_built_in_tools_params: StandardBuiltInToolsParams | None = None,
    ) -> tuple[ToolUse, tuple[Mapping[str, object], ...]]:
        """The built-in tools a call used and the rows their calculators read, for Rust to price. Each row is looked up
        only when a tool that reads it was used."""
        from litellm.llms.gemini.cost_calculator import google_maps_grounding_requests

        params: Final[Mapping[str, object]] = cast(  # cast-ok: a TypedDict is a str-keyed mapping
            Mapping[str, object], standard_built_in_tools_params or {}
        )
        searched: Final = StandardBuiltInToolCostTracking.response_object_includes_web_search_call(
            response_object=response_object, usage=usage
        )
        searched_files: Final = StandardBuiltInToolCostTracking.response_object_includes_file_search_call(
            response_object=response_object
        )
        maps_requests: Final = _requests(google_maps_grounding_requests(usage)) if usage is not None else None
        model_info, provider = (
            StandardBuiltInToolCostTracking._resolve_model_info(model=model, custom_llm_provider=custom_llm_provider)
            if searched or maps_requests is not None
            else (None, custom_llm_provider)
        )
        search: Final = (
            StandardBuiltInToolCostTracking._web_search_facts(model_info, provider, usage, response_object, params)
            if searched
            else None
        )
        code_sessions: Final = StandardBuiltInToolCostTracking._safe_convert_to_int(
            params.get("code_interpreter_sessions")
        )
        reads_direct: Final = searched_files or any(
            params.get(key) for key in ("vector_store_usage", "computer_use_usage", "code_interpreter_sessions")
        )
        direct: Final = (
            StandardBuiltInToolCostTracking._safe_get_model_info(model=model, custom_llm_provider=custom_llm_provider)
            if reads_direct
            else None
        )
        image: Final = StandardBuiltInToolCostTracking._image_generation_facts(response_object, custom_llm_provider)
        positions, rows = _rows_at(
            (
                model_info,
                None if search is None else search.served,
                direct,
                _container_row(custom_llm_provider) if code_sessions else None,
                None if image is None else image.row,
                *(() if image is None else image.item_rows),
            )
        )
        model_at, served_at, direct_at, container_at, image_at, *items_at = positions
        file_search: Final = params.get("file_search") or None
        storage_gb, days = StandardBuiltInToolCostTracking._extract_file_search_params(file_search)
        return (
            ToolUse(
                provider=custom_llm_provider,
                maps_grounding=None
                if maps_requests is None or model_info is None or provider is None
                else MapsGroundingTool(provider=provider, row=model_at, requests=maps_requests),
                image_generation=None if image is None else image.tool(image_at, items_at),
                web_search=None if search is None else search.tool(model_at, served_at),
                file_search=None
                if not searched_files
                else FileSearchUse(
                    requested=file_search is not None,
                    provider=custom_llm_provider,
                    row=direct_at,
                    storage_gb=storage_gb,
                    days=days,
                    per_call=OPENAI_FILE_SEARCH_COST_PER_1K_CALLS,
                    azure_gigabyte_day=AZURE_FILE_SEARCH_COST_PER_GB_PER_DAY,
                ),
                vector_store=StandardBuiltInToolCostTracking._vector_store_facts(
                    params.get("vector_store_usage"), custom_llm_provider, direct_at
                ),
                computer_use=StandardBuiltInToolCostTracking._computer_use_facts(
                    params.get("computer_use_usage"), custom_llm_provider, direct_at
                ),
                code_interpreter=None
                if not code_sessions
                else CodeInterpreterTool(row=direct_at, container_row=container_at, sessions=max(code_sessions, 0)),
            ),
            rows,
        )

    @staticmethod
    def _web_search_facts(
        model_info: ModelInfo | None,
        provider: str | None,
        usage: Usage | None,
        response_object: object,
        params: Mapping[str, object],
    ) -> "_WebSearchFacts":
        resolved_usage: Final = StandardBuiltInToolCostTracking._usage_with_anthropic_web_search(
            usage=usage, response_object=response_object
        )
        response_model: Final = _output_item_field(response_object, "model")
        dispatched: Final = (
            web_search_dispatch(
                provider,
                resolved_usage,
                model_info,
                response_model if isinstance(response_model, str) else None,
                StandardBuiltInToolCostTracking._safe_get_model_info,
            )
            if model_info is not None and resolved_usage is not None and provider is not None
            else None
        )
        options: Final[object] = params.get("web_search_options")
        size: Final[object] = options.get("search_context_size") if isinstance(options, Mapping) else None
        served_at: Final = None if dispatched is None else dispatched[0]["served_row"]
        return _WebSearchFacts(
            dispatch=None if dispatched is None else dispatched[0],
            served=None
            if dispatched is None or served_at is None
            else cast(ModelInfo, dispatched[1][served_at]),  # cast-ok: the served row is a ModelInfo
            priced_per_call=model_info is not None,
            context=_search_context(size),
            calls=StandardBuiltInToolCostTracking._billable_web_search_calls(resolved_usage, response_object),
        )

    @staticmethod
    def _image_generation_facts(
        response_object: object, custom_llm_provider: str | None
    ) -> "_ImageGenerationFacts | None":
        if not isinstance(response_object, ResponsesAPIResponse):
            return None
        output: Final[Sequence[object]] = cast(Sequence[object], response_object.output)  # cast-ok: isinstance-narrowed
        completed_items: Final = tuple(
            output_item
            for output_item in output
            if _output_item_type(output_item) == "image_generation_call"
            and _output_item_field(output_item, "status") == "completed"
        )
        if not completed_items:
            return None
        tool_model: Final = StandardBuiltInToolCostTracking._image_generation_tool_model(response_object)
        if tool_model is None:
            verbose_logger.debug("Responses API image_generation_call has no tools[].model to price, billing 0.0")
            return None
        tool_usage: Final = _responses_tool_usage(response_object)
        image_gen: Final = None if tool_usage is None else tool_usage.image_gen
        input_details: Final = None if image_gen is None else image_gen.input_tokens_details
        output_details: Final = None if image_gen is None else image_gen.output_tokens_details
        tokens: Final = (
            None
            if image_gen is None or image_gen.total_tokens <= 0
            else ImageTokens(
                text_input=input_details.text_tokens if input_details else 0,
                image_input=input_details.image_tokens if input_details else 0,
                image_output=output_details.image_tokens if output_details else 0,
                text_output=output_details.text_tokens if output_details else 0,
            )
        )
        items: Final = tuple(
            _image_item(output_item, tool_model, custom_llm_provider) for output_item in completed_items
        )
        return _ImageGenerationFacts(
            tokens=tokens,
            row=None
            if tokens is None
            else StandardBuiltInToolCostTracking._safe_get_model_info(
                model=tool_model, custom_llm_provider=custom_llm_provider
            ),
            item_rows=tuple(row for row, _ in items),
            pixels=tuple(pixels for _, pixels in items),
        )

    @staticmethod
    def _vector_store_facts(usage: object, provider: str | None, row: int | None) -> VectorStoreTool | None:
        if not usage:
            return None
        fields: Final[Mapping[str, object]] = (
            cast(Mapping[str, object], usage) if isinstance(usage, dict) else {}  # cast-ok: narrowed by isinstance
        )
        return VectorStoreTool(
            provider=provider,
            row=row,
            storage_gb=_float(fields.get("storage_gb", 0.0)) or 0.0,
            days=_float(fields.get("days", 0.0)) or 0.0,
            azure_gigabyte_day=AZURE_VECTOR_STORE_COST_PER_GB_PER_DAY,
        )

    @staticmethod
    def _computer_use_facts(usage: object, provider: str | None, row: int | None) -> ComputerUseTool | None:
        if not usage:
            return None
        input_tokens, output_tokens = StandardBuiltInToolCostTracking._extract_token_counts(usage)
        return ComputerUseTool(
            provider=provider,
            row=row,
            input_tokens=max(input_tokens or 0, 0),
            output_tokens=max(output_tokens or 0, 0),
            azure_input=AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS,
            azure_output=AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS,
        )

    @staticmethod
    def _resolve_model_info(model: str, custom_llm_provider: str | None) -> tuple[ModelInfo | None, str | None]:
        direct: Final = StandardBuiltInToolCostTracking._safe_get_model_info(
            model=model, custom_llm_provider=custom_llm_provider
        )
        if direct is not None:
            return direct, custom_llm_provider or direct["litellm_provider"]
        if "/" not in model:
            return None, custom_llm_provider
        by_prefix: Final = StandardBuiltInToolCostTracking._safe_get_model_info(model=model)
        if by_prefix is None:
            return None, custom_llm_provider
        return by_prefix, by_prefix["litellm_provider"]

    @staticmethod
    def _billable_web_search_calls(usage: Usage | None, response_object: object) -> int:
        """
        Providers without their own calculator but whose usage carries server_tool_use.web_search_requests
        (OpenAI-compatible providers that report a per-tool count) are billed that many calls on the per-call path.
        """
        reported: Final = None if usage is None else get_web_search_requests_from_usage(usage)
        if isinstance(reported, int) and not isinstance(reported, bool) and reported > 0:
            return reported
        return StandardBuiltInToolCostTracking._count_web_search_calls(response_object)

    @staticmethod
    def _count_web_search_calls(response_object: object) -> int:
        """
        Number of web searches to bill for on the per-call pricing path.

        Providers that report a request count in usage (gemini, anthropic, xai, vertex) are handled by
        get_cost_for_web_search_request and never reach here. This path prices per call, so it must count
        the web_search_call items, unless the response reports the billable count itself
        (Bedrock's tool_usage.web_search.num_requests, which excludes open_page fetches). Chat-completions
        responses only expose url_citation annotations with no count, so they floor to a single billable search.
        """
        if not isinstance(response_object, ResponsesAPIResponse):
            return 1
        reported: Final = _reported_web_search_requests(response_object)
        if reported is not None:
            return max(reported, 0)
        count: Final = sum(
            1 for output_item in response_object.output if _output_item_type(output_item) == "web_search_call"
        )
        return max(count, 1)

    @staticmethod
    def _image_generation_tool_model(response_object: ResponsesAPIResponse) -> str | None:
        tools: Final[Sequence[object]] = cast(Sequence[object], response_object.tools or ())  # cast-ok: dict tool union
        for tool in tools:
            if _output_item_field(tool, "type") != "image_generation":
                continue
            if isinstance(model := _output_item_field(tool, "model"), str) and model:
                return model
        return None

    @staticmethod
    def _extract_file_search_params(
        file_search_usage: object,
    ) -> tuple[float | None, float | None]:
        """Extract and convert file search parameters safely."""
        if not isinstance(file_search_usage, dict):
            return None, None
        fields: Final[Mapping[str, object]] = cast(  # cast-ok: narrowed by isinstance
            Mapping[str, object], file_search_usage
        )
        storage_gb: Final[object] = fields.get("storage_gb")
        days: Final[object] = fields.get("days")
        return (
            None if storage_gb is None else _float(storage_gb),
            None if days is None else _float(days),
        )

    @staticmethod
    def _extract_token_counts(
        computer_use_usage: object,
    ) -> tuple[int | None, int | None]:
        """Extract and convert token counts safely."""
        if not isinstance(computer_use_usage, dict):
            return None, None
        fields: Final[Mapping[str, object]] = cast(  # cast-ok: narrowed by isinstance
            Mapping[str, object], computer_use_usage
        )
        return (
            StandardBuiltInToolCostTracking._safe_convert_to_int(fields.get("input_tokens")),
            StandardBuiltInToolCostTracking._safe_convert_to_int(fields.get("output_tokens")),
        )

    @staticmethod
    def _safe_convert_to_int(value: object) -> int | None:
        """Safely convert a value to int."""
        if isinstance(value, (int, float, str)):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _usage_with_anthropic_web_search(usage: Usage | None, response_object: object) -> Usage | None:
        """Return a Usage carrying server_tool_use.web_search_requests sourced from a
        raw Anthropic /v1/messages response dict when the reconstructed Usage dropped
        it (or was never supplied). The original Usage is returned unchanged when it
        already exposes the field or the response is not an Anthropic dict."""
        from litellm.llms.anthropic.cost_calculation import (
            get_anthropic_web_search_requests_from_response,
        )

        if usage is not None and (get_web_search_requests_from_usage(usage) is not None):
            return usage
        web_search_requests: Final = get_anthropic_web_search_requests_from_response(response_object)
        if web_search_requests is None:
            return usage
        server_tool_use: Final = ServerToolUse(web_search_requests=web_search_requests)
        if usage is None:
            return Usage(server_tool_use=server_tool_use)
        return usage.model_copy(update={"server_tool_use": server_tool_use})

    @staticmethod
    def response_object_includes_web_search_call(response_object: object, usage: Usage | None = None) -> bool:
        """
        Check if the response object includes a web search call.

        This covers:
        - Chat Completion Response (ModelResponse)
        - ResponsesAPIResponse (streaming + non-streaming)
        - Anthropic /v1/messages raw response dict
        """
        from litellm.llms.anthropic.cost_calculation import (
            get_anthropic_web_search_requests_from_response,
        )

        if get_anthropic_web_search_requests_from_response(response_object) is not None:
            return True

        if isinstance(response_object, ModelResponse):
            # chat completions only include url_citation annotations when a web search call is made
            has_url_citations: Final = StandardBuiltInToolCostTracking.response_includes_annotation_type(
                response_object=response_object, annotation_type="url_citation"
            )
            if has_url_citations:
                return True
            if usage is not None:
                # Vertex AI Gemini uses usage.prompt_tokens_details.web_search_requests
                if (
                    hasattr(usage, "prompt_tokens_details")
                    and usage.prompt_tokens_details is not None
                    and isinstance(usage.prompt_tokens_details, PromptTokensDetailsWrapper)
                    and hasattr(usage.prompt_tokens_details, "web_search_requests")
                    and usage.prompt_tokens_details.web_search_requests is not None
                ):
                    return True
                # Anthropic Claude (direct API and Vertex AI) uses server_tool_use.web_search_requests.
                # Without this check, Claude ModelResponse always falls through to return False
                # and _handle_web_search_cost() is never called.
                if get_web_search_requests_from_usage(usage) is not None:
                    return True
                # xAI reports usage.server_side_tool_usage_details.web_search_calls; a searched
                # answer with no url_citation annotations has no other chat-path signal
                if _usage_reports_server_side_web_search_calls(usage):
                    return True
            return False
        elif isinstance(response_object, ResponsesAPIResponse):
            # response api explicitly includes web_search_call in the output
            if StandardBuiltInToolCostTracking.response_includes_output_type(
                response_object=response_object, output_type="web_search_call"
            ):
                return True
            reported: Final = _reported_web_search_requests(response_object)
            return reported is not None and reported > 0
        elif usage is not None:
            if get_web_search_requests_from_usage(usage) is not None or (
                hasattr(usage, "prompt_tokens_details")
                and usage.prompt_tokens_details is not None
                and isinstance(usage.prompt_tokens_details, PromptTokensDetailsWrapper)
                and hasattr(usage.prompt_tokens_details, "web_search_requests")
                and usage.prompt_tokens_details.web_search_requests is not None
            ):
                return True
            if _usage_reports_server_side_web_search_calls(usage):
                return True

        return False

    @staticmethod
    def response_object_includes_file_search_call(
        response_object: object,
    ) -> bool:
        """
        Check if the response object includes a file search call.

        This covers:
            - Chat Completion Response (ModelResponse)
            - ResponsesAPIResponse (streaming + non-streaming)
        """
        if isinstance(response_object, ModelResponse):
            # chat completions only include file_citation annotations when a file search call is made
            return StandardBuiltInToolCostTracking.response_includes_annotation_type(
                response_object=response_object, annotation_type="file_citation"
            )
        elif isinstance(response_object, ResponsesAPIResponse):
            # response api explicitly includes file_search_call in the output
            return StandardBuiltInToolCostTracking.response_includes_output_type(
                response_object=response_object, output_type="file_search_call"
            )
        return False

    @staticmethod
    def response_includes_annotation_type(
        response_object: ModelResponse,
        annotation_type: Literal["url_citation", "file_citation"],
    ) -> bool:
        if isinstance(response_object, ModelResponse):
            for choice in response_object.choices:
                message: Message | None = getattr(choice, "message", None)
                if message is None:
                    continue
                annotations: list[ChatCompletionAnnotation] | None = getattr(message, "annotations", None)
                if annotations:
                    for annotation in annotations:
                        if annotation.get("type", None) == annotation_type:
                            return True
        return False

    @staticmethod
    def response_includes_output_type(
        response_object: ResponsesAPIResponse,
        output_type: Literal["web_search_call", "file_search_call"],
    ) -> bool:
        """
        Check if the ResponsesAPIResponse includes one of the specified output types.

        This is used for cost tracking of built-in tools.

        Args:
            response_object: The ResponsesAPIResponse object to check.
            output_type: The type of output to check for.

        Returns:
            True if the ResponsesAPIResponse includes one of the specified output types, False otherwise.
        """
        return any(_output_item_type(output_item) == output_type for output_item in response_object.output)

    @staticmethod
    def _safe_get_model_info(model: str, custom_llm_provider: str | None = None) -> ModelInfo | None:
        try:
            return litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
        except Exception:
            return None

    @staticmethod
    def get_cost_for_web_search(
        web_search_options: WebSearchOptions | None = None,
        model_info: ModelInfo | None = None,
    ) -> float:
        """
        If request includes `web_search_options`, calculate the cost of the web search.
        """
        options: Final[Mapping[str, object]] = cast(  # cast-ok: a TypedDict is a str-keyed mapping
            Mapping[str, object], web_search_options or {}
        )
        return StandardBuiltInToolCostTracking._price_per_call(
            model_info, _search_context(options.get("search_context_size"))
        )

    @staticmethod
    def get_default_cost_for_web_search(
        model_info: ModelInfo | None = None,
    ) -> float:
        """
        If no web search options are provided, use the `search_context_size_medium` pricing.

        https://platform.openai.com/docs/pricing#web-search
        """
        return StandardBuiltInToolCostTracking._price_per_call(model_info, None)

    @staticmethod
    def _price_per_call(model_info: ModelInfo | None, context: SearchContextSize | None) -> float:
        rows: Final[tuple[Mapping[str, object], ...]] = _NO_ROWS if model_info is None else (model_info,)
        per_call: Final = WebSearchPerCall(row=None if model_info is None else 0, context=context, calls=1)
        return (
            price_tools(OneToolOnly(one=OneWebSearch(tool="web_search", dispatch=None, per_call=per_call)), rows) or 0.0
        )

    @staticmethod
    def get_cost_for_file_search(
        file_search: FileSearchTool | None = None,
        provider: str | None = None,
        model_info: Mapping[str, object] | None = None,
        storage_gb: float | None = None,
        days: float | None = None,
    ) -> float:
        """
        OpenAI bills per call, Azure by storage (GB per day).

        Doc: https://platform.openai.com/docs/pricing#built-in-tools
        """
        files: Final = OneFileSearch(
            tool="file_search",
            **FileSearchUse(
                requested=file_search is not None,
                provider=provider,
                row=0 if model_info else None,
                storage_gb=storage_gb,
                days=days,
                per_call=OPENAI_FILE_SEARCH_COST_PER_1K_CALLS,
                azure_gigabyte_day=AZURE_FILE_SEARCH_COST_PER_GB_PER_DAY,
            ),
        )
        return price_tools(OneToolOnly(one=files), (model_info,) if model_info else _NO_ROWS) or 0.0

    @staticmethod
    def get_cost_for_vector_store(
        vector_store_usage: Mapping[str, object] | None = None,
        provider: str | None = None,
        model_info: Mapping[str, object] | None = None,
    ) -> float:
        """
        Calculate cost for vector store usage.

        Azure charges based on storage size and duration.
        """
        store: Final = StandardBuiltInToolCostTracking._vector_store_facts(
            vector_store_usage, provider, 0 if model_info else None
        )
        if store is None:
            return 0.0
        return (
            price_tools(
                OneToolOnly(one=OneVectorStore(tool="vector_store", **store)),
                (model_info,) if model_info else _NO_ROWS,
            )
            or 0.0
        )

    @staticmethod
    def get_cost_for_computer_use(
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        provider: str | None = None,
        model_info: Mapping[str, object] | None = None,
    ) -> float:
        """
        Calculate cost for computer use feature.

        Azure: $0.003 USD per 1K input tokens, $0.012 USD per 1K output tokens
        """
        return (
            price_tools(
                OneToolOnly(
                    one=OneComputerUse(
                        tool="computer_use",
                        provider=provider,
                        row=None if not model_info else 0,
                        input_tokens=max(input_tokens or 0, 0),
                        output_tokens=max(output_tokens or 0, 0),
                        azure_input=AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS,
                        azure_output=AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS,
                    )
                ),
                _NO_ROWS if not model_info else (model_info,),
            )
            or 0.0
        )

    @staticmethod
    def get_cost_for_code_interpreter(
        sessions: int | None = None,
        provider: str | None = None,
        model_info: Mapping[str, object] | None = None,
    ) -> float:
        """
        Calculate cost for code interpreter feature: the sessions at the row's price, else at the provider's
        ``<provider>/container`` row's.
        """
        if not sessions:
            return 0.0
        (row, container_row), rows = _rows_at((model_info or None, _container_row(provider)))
        return (
            price_tools(
                OneToolOnly(
                    one=OneCodeInterpreter(
                        tool="code_interpreter", row=row, container_row=container_row, sessions=max(sessions, 0)
                    )
                ),
                rows,
            )
            or 0.0
        )

    @staticmethod
    def chat_completion_response_includes_annotations(
        response_object: ModelResponse,
    ) -> bool:
        for _choice in response_object.choices:
            message: Message | None = getattr(_choice, "message", None)
            if (
                message is not None
                and hasattr(message, "annotations")
                and message.annotations is not None
                and len(message.annotations) > 0
            ):
                return True
        return False

    @staticmethod
    def get_web_search_options(kwargs: Mapping[str, object]) -> WebSearchOptions | None:
        web_search_options: Final = kwargs.get("web_search_options")
        if isinstance(web_search_options, dict):
            return WebSearchOptions(**cast(WebSearchOptions, web_search_options))

        tools: Final = StandardBuiltInToolCostTracking._get_tools_from_kwargs(
            kwargs=kwargs, tool_type="web_search_preview"
        ) or StandardBuiltInToolCostTracking._get_tools_from_kwargs(kwargs=kwargs, tool_type="web_search")
        if tools:
            # Look for web search tool in the tools array
            for tool in tools:
                if isinstance(tool, dict):
                    if StandardBuiltInToolCostTracking._is_web_search_tool_call(tool):
                        return WebSearchOptions(**cast(WebSearchOptions, tool))
        return None

    _get_web_search_options = get_web_search_options

    @staticmethod
    def _get_tools_from_kwargs(kwargs: Mapping[str, object], tool_type: str) -> Sequence[object] | None:
        tools: Final = kwargs.get("tools")
        return cast(Sequence[object], tools) if isinstance(tools, (list, tuple)) else None

    @staticmethod
    def get_file_search_tool_call(kwargs: Mapping[str, object]) -> FileSearchTool | None:
        tools: Final = StandardBuiltInToolCostTracking._get_tools_from_kwargs(kwargs, "file_search")
        if tools:
            for tool in tools:
                if isinstance(tool, dict):
                    if StandardBuiltInToolCostTracking._is_file_search_tool_call(tool):
                        return FileSearchTool(**cast(FileSearchTool, tool))
        return None

    _get_file_search_tool_call = get_file_search_tool_call

    @staticmethod
    def _is_web_search_tool_call(tool: Mapping[str, object]) -> bool:
        if tool.get("type", None) == "web_search_preview":
            return True
        if tool.get("type", None) == "web_search":
            return True
        if "search_context_size" in tool:
            return True
        return False

    @staticmethod
    def _is_file_search_tool_call(tool: Mapping[str, object]) -> bool:
        if tool.get("type", None) == "file_search":
            return True
        return False


@dataclass(frozen=True, slots=True)
class _WebSearchFacts:
    """What ``_handle_web_search_cost`` read: the dispatch facts when it had a row, usage and provider to dispatch
    with, the served model's row it may borrow search prices from, and the calls it bills per call otherwise."""

    dispatch: WebSearchDispatch | None
    served: ModelInfo | None
    priced_per_call: bool
    context: SearchContextSize | None
    calls: int

    def tool(self, model_at: int | None, served_at: int | None) -> WebSearchTool:
        return WebSearchTool(
            dispatch=None
            if self.dispatch is None
            else WebSearchDispatch(
                provider=self.dispatch["provider"],
                row=model_at,
                served_row=served_at,
                model_key=self.dispatch["model_key"],
                response_model=self.dispatch["response_model"],
                counts=self.dispatch["counts"],
            ),
            per_call=WebSearchPerCall(
                row=model_at if self.priced_per_call else None, context=self.context, calls=self.calls
            ),
        )


@dataclass(frozen=True, slots=True)
class _ImageGenerationFacts:
    """The completed image generation calls of a Responses API response: the tool's reported tokens and its model's
    row, and each call's row and pixels as ``default_image_cost_calculator`` reads them."""

    tokens: ImageTokens | None
    row: ModelInfo | None
    item_rows: tuple[Mapping[str, object] | None, ...]
    pixels: tuple[int, ...]

    def tool(self, row_at: int | None, items_at: Sequence[int | None]) -> ImageGenerationTool:
        return ImageGenerationTool(
            row=row_at,
            tokens=self.tokens,
            items=tuple(
                None if at is None else ImageItem(row=at, pixels=pixels) for at, pixels in zip(items_at, self.pixels)
            ),
        )


def _image_item(
    output_item: object, tool_model: str, custom_llm_provider: str | None
) -> tuple[Mapping[str, object] | None, int]:
    """The row and pixels ``default_image_cost_calculator`` reads for one image generation call, no row where it
    raised."""
    from litellm.cost_calculator import default_image_row_key

    quality: Final = _output_item_field(output_item, "quality")
    size: Final = _output_item_field(output_item, "size")
    try:
        key, pixels, _ = default_image_row_key(
            tool_model,
            custom_llm_provider,
            quality if isinstance(quality, str) and quality != "auto" else None,
            size if isinstance(size, str) and size != "auto" else None,
        )
    except ValueError as e:
        verbose_logger.debug("Could not price Responses API image_generation_call item: %s", e)
        return None, 0
    row: Final[object] = None if key is None else litellm.model_cost.get(key)  # pyright: ignore[reportUnknownMemberType]  # model cost map is dynamically loaded
    found: Final = cast(Mapping[str, object], row) if isinstance(row, dict) else None  # cast-ok: narrowed by isinstance
    return found, pixels
