"""
Helper utilities for tracking the cost of built-in tools.
"""

from typing import Any, Dict, List, Literal, Optional

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

        """
        from litellm.llms import get_cost_for_web_search_request

        standard_built_in_tools_params = standard_built_in_tools_params or {}
        #########################################################
        # Web Search
        #########################################################
        if StandardBuiltInToolCostTracking.response_object_includes_web_search_call(
            response_object=response_object,
            usage=usage,
        ):
            model_info = StandardBuiltInToolCostTracking._safe_get_model_info(
                model=model, custom_llm_provider=custom_llm_provider
            )
            result: Optional[float] = None
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
            if result is None:
                return StandardBuiltInToolCostTracking.get_cost_for_web_search(
                    web_search_options=standard_built_in_tools_params.get(
                        "web_search_options", None
                    ),
                    model_info=model_info,
                )
            else:
                return result

        #########################################################
        # File Search
        #########################################################
        elif StandardBuiltInToolCostTracking.response_object_includes_file_search_call(
            response_object=response_object
        ):
            return StandardBuiltInToolCostTracking.get_cost_for_file_search(
                file_search=standard_built_in_tools_params.get("file_search", None),
            )

        return 0.0

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

        search_context_pricing: SearchContextCostPerQuery = (
            model_info.get("search_context_cost_per_query", {}) or {}
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
        search_context_pricing: SearchContextCostPerQuery = (
            model_info.get("search_context_cost_per_query", {}) or {}
        ) or {}
        return search_context_pricing.get("search_context_size_medium", 0.0)

    @staticmethod
    def get_cost_for_file_search(
        file_search: Optional[FileSearchTool] = None,
    ) -> float:
        """ "
        Charged at $2.50/1k calls

        Doc: https://platform.openai.com/docs/pricing#built-in-tools
        """
        if file_search is None:
            return 0.0
        return OPENAI_FILE_SEARCH_COST_PER_1K_CALLS

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
            kwargs, "web_search_preview"
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
            tools = kwargs.get("tools", [])
            return tools
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
        if "search_context_size" in tool:
            return True
        return False

    @staticmethod
    def _is_file_search_tool_call(tool: Dict) -> bool:
        if tool.get("type", None) == "file_search":
            return True
        return False
