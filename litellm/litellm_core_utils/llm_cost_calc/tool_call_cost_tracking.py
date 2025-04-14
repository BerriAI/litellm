"""
Helper utilities for tracking the cost of built-in tools.
"""

from typing import Any, Dict, List, Optional

import litellm
from litellm.constants import OPENAI_FILE_SEARCH_COST_PER_1K_CALLS
from litellm.types.llms.openai import FileSearchTool, WebSearchOptions
from litellm.types.utils import (
    ModelInfo,
    ModelResponse,
    SearchContextCostPerQuery,
    StandardBuiltInToolsParams,
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
        custom_llm_provider: Optional[str] = None,
        standard_built_in_tools_params: Optional[StandardBuiltInToolsParams] = None,
    ) -> float:
        """
        Get the cost of using built-in tools.

        Supported tools:
        - Web Search

        """
        if standard_built_in_tools_params is not None:
            if (
                standard_built_in_tools_params.get("web_search_options", None)
                is not None
            ):
                model_info = StandardBuiltInToolCostTracking._safe_get_model_info(
                    model=model, custom_llm_provider=custom_llm_provider
                )

                return StandardBuiltInToolCostTracking.get_cost_for_web_search(
                    web_search_options=standard_built_in_tools_params.get(
                        "web_search_options", None
                    ),
                    model_info=model_info,
                )

            if standard_built_in_tools_params.get("file_search", None) is not None:
                return StandardBuiltInToolCostTracking.get_cost_for_file_search(
                    file_search=standard_built_in_tools_params.get("file_search", None),
                )

        if isinstance(response_object, ModelResponse):
            if StandardBuiltInToolCostTracking.chat_completion_response_includes_annotations(
                response_object
            ):
                model_info = StandardBuiltInToolCostTracking._safe_get_model_info(
                    model=model, custom_llm_provider=custom_llm_provider
                )
                return StandardBuiltInToolCostTracking.get_default_cost_for_web_search(
                    model_info
                )
        return 0.0

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
        if web_search_options is None:
            return 0.0
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
