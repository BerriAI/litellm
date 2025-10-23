"""
Helper utilities for tracking the cost of built-in tools.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

import litellm
from litellm.constants import OPENAI_FILE_SEARCH_COST_PER_1K_CALLS
from litellm.types.llms.openai import (
    FileSearchTool,
    ResponsesAPIResponse,
    WebSearchOptions,
)
from litellm.types.utils import (
    Message,
    ModelInfo,
    ModelResponse,
    SearchContextCostPerQuery,
    StandardBuiltInToolsParams,
    Usage,
)


class StandardBuiltInToolCostTracking:
    """
    Helper class for tracking the cost of built-in tools

    Example: Web Search
    """

    @staticmethod
    def get_cost_for_built_in_tools(
        model: str,
        response_object: Any,
        usage: Optional[Usage] = None,
        custom_llm_provider: Optional[str] = None,
        standard_built_in_tools_params: Optional[StandardBuiltInToolsParams] = None,
    ) -> float:
        """
        Get the cost of using built-in tools.

        Supported tools:
        - Web Search
        - File Search
        - Vector Store (Azure)
        - Computer Use (Azure)
        - Code Interpreter (Azure)
        """
        standard_built_in_tools_params = standard_built_in_tools_params or {}

        # Handle web search
        if StandardBuiltInToolCostTracking.response_object_includes_web_search_call(
            response_object=response_object, usage=usage
        ):
            return StandardBuiltInToolCostTracking._handle_web_search_cost(
                model=model,
                custom_llm_provider=custom_llm_provider,
                usage=usage,
                standard_built_in_tools_params=standard_built_in_tools_params,
            )

        # Handle file search
        if StandardBuiltInToolCostTracking.response_object_includes_file_search_call(
            response_object=response_object
        ):
            return StandardBuiltInToolCostTracking._handle_file_search_cost(
                model=model,
                custom_llm_provider=custom_llm_provider,
                standard_built_in_tools_params=standard_built_in_tools_params,
            )

        # Handle Azure assistant features
        return StandardBuiltInToolCostTracking._handle_azure_assistant_costs(
            model=model,
            custom_llm_provider=custom_llm_provider,
            standard_built_in_tools_params=standard_built_in_tools_params,
        )

    @staticmethod
    def _handle_web_search_cost(
        model: str,
        custom_llm_provider: Optional[str],
        usage: Optional[Usage],
        standard_built_in_tools_params: StandardBuiltInToolsParams,
    ) -> float:
        """Handle web search cost calculation."""
        from litellm.llms import get_cost_for_web_search_request

        model_info = StandardBuiltInToolCostTracking._safe_get_model_info(
            model=model, custom_llm_provider=custom_llm_provider
        )

        if custom_llm_provider is None and model_info is not None:
            custom_llm_provider = model_info["litellm_provider"]

        if (
            model_info is not None
            and usage is not None
            and custom_llm_provider is not None
        ):
            result = get_cost_for_web_search_request(
                custom_llm_provider=custom_llm_provider,
                usage=usage,
                model_info=model_info,
            )
            if result is not None:
                return result

        return StandardBuiltInToolCostTracking.get_cost_for_web_search(
            web_search_options=standard_built_in_tools_params.get(
                "web_search_options", None
            ),
            model_info=model_info,
        )

    @staticmethod
    def _handle_file_search_cost(
        model: str,
        custom_llm_provider: Optional[str],
        standard_built_in_tools_params: StandardBuiltInToolsParams,
    ) -> float:
        """Handle file search cost calculation."""
        model_info = StandardBuiltInToolCostTracking._safe_get_model_info(
            model=model, custom_llm_provider=custom_llm_provider
        )
        file_search_raw: Any = standard_built_in_tools_params.get("file_search", {})
        file_search_usage: Optional[FileSearchTool] = (
            FileSearchTool(**file_search_raw) if file_search_raw else None
        )

        # Convert model_info to dict and extract usage parameters
        model_info_dict = dict(model_info) if model_info is not None else None
        storage_gb, days = StandardBuiltInToolCostTracking._extract_file_search_params(
            file_search_usage
        )

        return StandardBuiltInToolCostTracking.get_cost_for_file_search(
            file_search=file_search_usage,
            provider=custom_llm_provider,
            model_info=model_info_dict,
            storage_gb=storage_gb,
            days=days,
        )

    @staticmethod
    def _handle_azure_assistant_costs(
        model: str,
        custom_llm_provider: Optional[str],
        standard_built_in_tools_params: StandardBuiltInToolsParams,
    ) -> float:
        """Handle Azure assistant features cost calculation."""
        if custom_llm_provider != "azure":
            return 0.0

        model_info = StandardBuiltInToolCostTracking._safe_get_model_info(
            model=model, custom_llm_provider=custom_llm_provider
        )

        total_cost = 0.0
        total_cost += StandardBuiltInToolCostTracking._get_vector_store_cost(
            model_info, custom_llm_provider, standard_built_in_tools_params
        )
        total_cost += StandardBuiltInToolCostTracking._get_computer_use_cost(
            model_info, custom_llm_provider, standard_built_in_tools_params
        )
        total_cost += StandardBuiltInToolCostTracking._get_code_interpreter_cost(
            model_info, custom_llm_provider, standard_built_in_tools_params
        )

        return total_cost

    @staticmethod
    def _extract_file_search_params(
        file_search_usage: Any,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Extract and convert file search parameters safely."""
        storage_gb = None
        days = None

        if isinstance(file_search_usage, dict):
            storage_gb_val = file_search_usage.get("storage_gb")
            days_val = file_search_usage.get("days")

            if storage_gb_val is not None:
                try:
                    storage_gb = float(storage_gb_val)  # type: ignore
                except (TypeError, ValueError):
                    storage_gb = None

            if days_val is not None:
                try:
                    days = float(days_val)  # type: ignore
                except (TypeError, ValueError):
                    days = None

        return storage_gb, days

    @staticmethod
    def _get_vector_store_cost(
        model_info: Optional[ModelInfo],
        custom_llm_provider: Optional[str],
        standard_built_in_tools_params: StandardBuiltInToolsParams,
    ) -> float:
        """Calculate vector store cost."""
        vector_store_usage = standard_built_in_tools_params.get(
            "vector_store_usage", None
        )
        if not vector_store_usage:
            return 0.0

        model_info_dict = dict(model_info) if model_info is not None else None
        vector_store_dict = (
            vector_store_usage if isinstance(vector_store_usage, dict) else {}
        )

        return StandardBuiltInToolCostTracking.get_cost_for_vector_store(
            vector_store_usage=vector_store_dict,
            provider=custom_llm_provider,
            model_info=model_info_dict,
        )

    @staticmethod
    def _get_computer_use_cost(
        model_info: Optional[ModelInfo],
        custom_llm_provider: Optional[str],
        standard_built_in_tools_params: StandardBuiltInToolsParams,
    ) -> float:
        """Calculate computer use cost."""
        computer_use_usage = standard_built_in_tools_params.get(
            "computer_use_usage", {}
        )
        if not computer_use_usage:
            return 0.0

        model_info_dict = dict(model_info) if model_info is not None else None
        input_tokens, output_tokens = (
            StandardBuiltInToolCostTracking._extract_token_counts(computer_use_usage)
        )

        return StandardBuiltInToolCostTracking.get_cost_for_computer_use(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider=custom_llm_provider,
            model_info=model_info_dict,
        )

    @staticmethod
    def _get_code_interpreter_cost(
        model_info: Optional[ModelInfo],
        custom_llm_provider: Optional[str],
        standard_built_in_tools_params: StandardBuiltInToolsParams,
    ) -> float:
        """Calculate code interpreter cost."""
        code_interpreter_sessions = standard_built_in_tools_params.get(
            "code_interpreter_sessions", None
        )
        if not code_interpreter_sessions:
            return 0.0

        model_info_dict = dict(model_info) if model_info is not None else None
        sessions = StandardBuiltInToolCostTracking._safe_convert_to_int(
            code_interpreter_sessions
        )

        return StandardBuiltInToolCostTracking.get_cost_for_code_interpreter(
            sessions=sessions,
            provider=custom_llm_provider,
            model_info=model_info_dict,
        )

    @staticmethod
    def _extract_token_counts(
        computer_use_usage: Any,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Extract and convert token counts safely."""
        input_tokens = None
        output_tokens = None

        if isinstance(computer_use_usage, dict):
            input_tokens_val = computer_use_usage.get("input_tokens")
            output_tokens_val = computer_use_usage.get("output_tokens")

            input_tokens = StandardBuiltInToolCostTracking._safe_convert_to_int(
                input_tokens_val
            )
            output_tokens = StandardBuiltInToolCostTracking._safe_convert_to_int(
                output_tokens_val
            )

        return input_tokens, output_tokens

    @staticmethod
    def _safe_convert_to_int(value: Any) -> Optional[int]:
        """Safely convert a value to int."""
        if value is not None:
            try:
                return int(value)  # type: ignore
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def response_object_includes_web_search_call(
        response_object: Any, usage: Optional[Usage] = None
    ) -> bool:
        """
        Check if the response object includes a web search call.

        This covers:
        - Chat Completion Response (ModelResponse)
        - ResponsesAPIResponse (streaming + non-streaming)
        """
        from litellm.types.utils import PromptTokensDetailsWrapper

        if isinstance(response_object, ModelResponse):
            # chat completions only include url_citation annotations when a web search call is made
            return StandardBuiltInToolCostTracking.response_includes_annotation_type(
                response_object=response_object, annotation_type="url_citation"
            )
        elif isinstance(response_object, ResponsesAPIResponse):
            # response api explicitly includes web_search_call in the output
            return StandardBuiltInToolCostTracking.response_includes_output_type(
                response_object=response_object, output_type="web_search_call"
            )
        elif usage is not None:
            if (
                hasattr(usage, "server_tool_use")
                and usage.server_tool_use is not None
                and usage.server_tool_use.web_search_requests is not None
            ):
                return True
            elif (
                hasattr(usage, "prompt_tokens_details")
                and usage.prompt_tokens_details is not None
                and isinstance(usage.prompt_tokens_details, PromptTokensDetailsWrapper)
                and hasattr(usage.prompt_tokens_details, "web_search_requests")
                and usage.prompt_tokens_details.web_search_requests is not None
            ):
                return True

        return False

    @staticmethod
    def response_object_includes_file_search_call(
        response_object: Any,
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
                message: Optional[Message] = getattr(choice, "message", None)
                if message is None:
                    continue
                if annotations := getattr(message, "annotations", None):
                    if len(annotations) > 0:
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
        output = response_object.output
        for output_item in output:
            _output_type: Optional[str] = getattr(output_item, "type", None)
            if _output_type == output_type:
                return True
        return False

    @staticmethod
    def _safe_get_model_info(
        model: str, custom_llm_provider: Optional[str] = None
    ) -> Optional[ModelInfo]:
        try:
            return litellm.get_model_info(
                model=model, custom_llm_provider=custom_llm_provider
            )
        except Exception:
            return None

    @staticmethod
    def get_cost_for_web_search(
        web_search_options: Optional[WebSearchOptions] = None,
        model_info: Optional[ModelInfo] = None,
    ) -> float:
        """
        If request includes `web_search_options`, calculate the cost of the web search.
        """
        web_search_options = web_search_options or {}
        if model_info is None:
            return 0.0

        search_context_raw: Any = model_info.get("search_context_cost_per_query", {})
        search_context_pricing: SearchContextCostPerQuery = (
            SearchContextCostPerQuery(**search_context_raw)
            if search_context_raw
            else SearchContextCostPerQuery()
        )
        if web_search_options.get("search_context_size", None) == "low":
            return search_context_pricing.get("search_context_size_low", 0.0)
        elif web_search_options.get("search_context_size", None) == "medium":
            return search_context_pricing.get("search_context_size_medium", 0.0)
        elif web_search_options.get("search_context_size", None) == "high":
            return search_context_pricing.get("search_context_size_high", 0.0)
        return StandardBuiltInToolCostTracking.get_default_cost_for_web_search(
            model_info
        )

    @staticmethod
    def get_default_cost_for_web_search(
        model_info: Optional[ModelInfo] = None,
    ) -> float:
        """
        If no web search options are provided, use the `search_context_size_medium` pricing.

        https://platform.openai.com/docs/pricing#web-search
        """
        if model_info is None:
            return 0.0
        search_context_raw: Any = model_info.get("search_context_cost_per_query", {}) or {}
        search_context_pricing: SearchContextCostPerQuery = (
            SearchContextCostPerQuery(**search_context_raw)
            if search_context_raw
            else SearchContextCostPerQuery()
        )
        return search_context_pricing.get("search_context_size_medium", 0.0)

    @staticmethod
    def get_cost_for_file_search(
        file_search: Optional[FileSearchTool] = None,
        provider: Optional[str] = None,
        model_info: Optional[dict] = None,
        storage_gb: Optional[float] = None,
        days: Optional[float] = None,
    ) -> float:
        """ "
        OpenAI: $2.50/1k calls
        Azure: $0.1 USD per 1 GB/Day (storage-based pricing)

        Doc: https://platform.openai.com/docs/pricing#built-in-tools
        """
        if file_search is None:
            return 0.0

        # Check if model-specific pricing is available
        if (
            model_info
            and "file_search_cost_per_gb_per_day" in model_info
            and provider == "azure"
        ):
            if storage_gb and days:
                return storage_gb * days * model_info["file_search_cost_per_gb_per_day"]
        elif model_info and "file_search_cost_per_1k_calls" in model_info:
            return model_info["file_search_cost_per_1k_calls"]

        # Azure has storage-based pricing for file search
        if provider == "azure":
            from litellm.constants import AZURE_FILE_SEARCH_COST_PER_GB_PER_DAY

            if storage_gb and days:
                return storage_gb * days * AZURE_FILE_SEARCH_COST_PER_GB_PER_DAY
            # Default to 0 if no storage info provided
            return 0.0

        # Default to OpenAI pricing (per-call based)
        return OPENAI_FILE_SEARCH_COST_PER_1K_CALLS

    @staticmethod
    def get_cost_for_vector_store(
        vector_store_usage: Optional[dict] = None,
        provider: Optional[str] = None,
        model_info: Optional[dict] = None,
    ) -> float:
        """
        Calculate cost for vector store usage.

        Azure charges based on storage size and duration.
        """
        if vector_store_usage is None:
            return 0.0

        storage_gb = vector_store_usage.get("storage_gb", 0.0)
        days = vector_store_usage.get("days", 0.0)

        # Check if model-specific pricing is available
        if model_info and "vector_store_cost_per_gb_per_day" in model_info:
            return storage_gb * days * model_info["vector_store_cost_per_gb_per_day"]

        # Azure has different pricing structure for vector store
        if provider == "azure":
            from litellm.constants import AZURE_VECTOR_STORE_COST_PER_GB_PER_DAY

            return storage_gb * days * AZURE_VECTOR_STORE_COST_PER_GB_PER_DAY

        # OpenAI doesn't charge separately for vector store (included in embeddings)
        return 0.0

    @staticmethod
    def get_cost_for_computer_use(
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        provider: Optional[str] = None,
        model_info: Optional[dict] = None,
    ) -> float:
        """
        Calculate cost for computer use feature.

        Azure: $0.003 USD per 1K input tokens, $0.012 USD per 1K output tokens
        """
        if provider == "azure" and (input_tokens or output_tokens):
            # Check if model-specific pricing is available
            if model_info:
                input_cost = model_info.get(
                    "computer_use_input_cost_per_1k_tokens", 0.0
                )
                output_cost = model_info.get(
                    "computer_use_output_cost_per_1k_tokens", 0.0
                )
                if input_cost or output_cost:
                    total_cost = 0.0
                    if input_tokens:
                        total_cost += (input_tokens / 1000.0) * input_cost
                    if output_tokens:
                        total_cost += (output_tokens / 1000.0) * output_cost
                    return total_cost

            # Azure default pricing
            from litellm.constants import (
                AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS,
                AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS,
            )

            total_cost = 0.0
            if input_tokens:
                total_cost += (
                    input_tokens / 1000.0
                ) * AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS
            if output_tokens:
                total_cost += (
                    output_tokens / 1000.0
                ) * AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS
            return total_cost

        # OpenAI doesn't charge separately for computer use yet
        return 0.0

    @staticmethod
    def get_cost_for_code_interpreter(
        sessions: Optional[int] = None,
        provider: Optional[str] = None,
        model_info: Optional[dict] = None,
    ) -> float:
        """
        Calculate cost for code interpreter feature.

        Azure: $0.03 USD per session
        """
        if sessions is None or sessions == 0:
            return 0.0

        # Check if model-specific pricing is available
        if model_info and "code_interpreter_cost_per_session" in model_info:
            return sessions * model_info["code_interpreter_cost_per_session"]

        # Azure pricing for code interpreter
        if provider == "azure":
            from litellm.constants import AZURE_CODE_INTERPRETER_COST_PER_SESSION

            return sessions * AZURE_CODE_INTERPRETER_COST_PER_SESSION

        # OpenAI doesn't charge separately for code interpreter yet
        return 0.0

    @staticmethod
    def chat_completion_response_includes_annotations(
        response_object: ModelResponse,
    ) -> bool:
        for _choice in response_object.choices:
            message = getattr(_choice, "message", None)
            if (
                message is not None
                and hasattr(message, "annotations")
                and message.annotations is not None
                and len(message.annotations) > 0
            ):
                return True
        return False

    @staticmethod
    def _get_web_search_options(kwargs: Dict) -> Optional[WebSearchOptions]:
        if "web_search_options" in kwargs:
            return WebSearchOptions(**kwargs.get("web_search_options", {}))

        tools = StandardBuiltInToolCostTracking._get_tools_from_kwargs(
            kwargs=kwargs, tool_type="web_search_preview"
        ) or StandardBuiltInToolCostTracking._get_tools_from_kwargs(
            kwargs=kwargs, tool_type="web_search"
        )
        if tools:
            # Look for web search tool in the tools array
            for tool in tools:
                if isinstance(tool, dict):
                    if StandardBuiltInToolCostTracking._is_web_search_tool_call(tool):
                        return WebSearchOptions(**tool)
        return None

    @staticmethod
    def _get_tools_from_kwargs(kwargs: Dict, tool_type: str) -> Optional[List[Dict]]:
        if "tools" in kwargs:
            return kwargs.get("tools", [])
        return None

    @staticmethod
    def _get_file_search_tool_call(kwargs: Dict) -> Optional[FileSearchTool]:
        tools = StandardBuiltInToolCostTracking._get_tools_from_kwargs(
            kwargs, "file_search"
        )
        if tools:
            for tool in tools:
                if isinstance(tool, dict):
                    if StandardBuiltInToolCostTracking._is_file_search_tool_call(tool):
                        return FileSearchTool(**tool)
        return None

    @staticmethod
    def _is_web_search_tool_call(tool: Dict) -> bool:
        if tool.get("type", None) == "web_search_preview":
            return True
        if tool.get("type", None) == "web_search":
            return True
        if "search_context_size" in tool:
            return True
        return False

    @staticmethod
    def _is_file_search_tool_call(tool: Dict) -> bool:
        if tool.get("type", None) == "file_search":
            return True
        return False
