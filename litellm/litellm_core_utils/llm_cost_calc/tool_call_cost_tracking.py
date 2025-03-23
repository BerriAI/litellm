"""
Helper utilities for tracking the cost of built-in tools.
"""

from typing import Any, Dict, Optional

import litellm
from litellm.types.llms.openai import WebSearchOptions
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
        model_info = litellm.get_model_info(
            model=model, custom_llm_provider=custom_llm_provider
        )

        if (
            standard_built_in_tools_params is not None
            and standard_built_in_tools_params.get("web_search_options", None)
            is not None
        ):
            return StandardBuiltInToolCostTracking.get_cost_for_web_search(
                web_search_options=standard_built_in_tools_params.get(
                    "web_search_options", None
                ),
                model_info=model_info,
            )

        if isinstance(response_object, ModelResponse):
            if StandardBuiltInToolCostTracking.chat_completion_response_includes_annotations(
                response_object
            ):
                return StandardBuiltInToolCostTracking.get_default_cost_for_web_search(
                    model_info
                )
        return 0.0

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
    def get_default_cost_for_web_search(model_info: ModelInfo) -> float:
        """
        If no web search options are provided, use the `search_context_size_medium` pricing.

        https://platform.openai.com/docs/pricing#web-search
        """
        search_context_pricing: SearchContextCostPerQuery = (
            model_info.get("search_context_cost_per_query", {}) or {}
        ) or {}
        return search_context_pricing.get("search_context_size_medium", 0.0)

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
        if "tools" in kwargs:
            tools = kwargs.get("tools", [])
            # Look for web search tool in the tools array
            for tool in tools:
                if isinstance(tool, dict):
                    if StandardBuiltInToolCostTracking._web_search_tool_call(tool):
                        return WebSearchOptions(**tool)
        return None

    @staticmethod
    def _web_search_tool_call(tool: Dict) -> bool:
        if tool.get("type", None) == "web_search_preview":
            return True
        if "search_context_size" in tool:
            return True
        return False
