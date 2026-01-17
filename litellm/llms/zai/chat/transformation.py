"""
Z.AI (ZhipuAI) GLM Chat Transformation Config

Supports GLM-4.7, GLM-4.6, GLM-4.5 and other Z.AI models with:
- Tool calling
- Reasoning/thinking modes
- Web search integration via tools

Web Search API Reference: https://docs.z.ai/api-reference/tools/web-search
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import litellm
from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

ZAI_API_BASE = "https://api.z.ai/api/paas/v4"


# Z.AI Web Search Tool TypedDict
class ZAIWebSearchConfig(TypedDict, total=False):
    """
    Z.AI web_search tool configuration.

    Reference: https://docs.z.ai/api-reference/llm/chat-completion
    """
    search_engine: Literal["search_pro_jina"]
    enable: bool
    search_query: Optional[str]
    count: int  # 1-50, default 10
    search_domain_filter: Optional[str]  # Whitelist domain
    search_recency_filter: Literal["oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"]
    content_size: Literal["medium", "high"]  # medium=400-600 chars, high=2500 chars
    result_sequence: Literal["before", "after"]  # When to show results
    search_result: bool  # Include raw results in response
    require_search: bool  # Force search-based response
    search_prompt: Optional[str]  # Custom search prompt


class ZAIWebSearchTool(TypedDict):
    """Z.AI web search tool format for tools array."""
    type: Literal["web_search"]
    web_search: ZAIWebSearchConfig


# Map OpenAI search_context_size to Z.AI count
ZAI_WEB_SEARCH_COUNT_MAP: Dict[str, int] = {
    "low": 5,
    "medium": 10,
    "high": 20,
}


class ZAIChatConfig(OpenAIGPTConfig):
    """
    Z.AI (ZhipuAI) GLM Chat Configuration.

    Extends OpenAIGPTConfig with Z.AI-specific features:
    - Web search via tools
    - Thinking/reasoning modes
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "zai"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("ZAI_API_BASE") or ZAI_API_BASE
        dynamic_api_key = api_key or get_secret_str("ZAI_API_KEY")
        return api_base, dynamic_api_key

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported OpenAI-compatible parameters for Z.AI models.

        Includes web_search_options for models that support web search.
        """
        base_params = [
            "max_tokens",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
        ]

        # Add web_search_options if model supports it
        try:
            if litellm.supports_web_search(
                model=model, custom_llm_provider=self.custom_llm_provider
            ):
                base_params.append("web_search_options")
        except Exception:
            pass

        # Add thinking/reasoning if model supports it
        try:
            if litellm.supports_reasoning(
                model=model, custom_llm_provider=self.custom_llm_provider
            ):
                base_params.append("thinking")
        except Exception:
            pass

        return base_params

    def _map_web_search_options(self, value: Dict[str, Any]) -> ZAIWebSearchTool:
        """
        Map OpenAI-style web_search_options to Z.AI web_search tool format.

        OpenAI format:
            {
                "search_context_size": "low" | "medium" | "high",
                "user_location": {...}  # Not supported by Z.AI
            }

        Z.AI format:
            {
                "type": "web_search",
                "web_search": {
                    "search_engine": "search_pro_jina",
                    "enable": true,
                    "count": 10,
                    "search_recency_filter": "noLimit",
                    "content_size": "medium"
                }
            }
        """
        # Get search context size, default to medium
        search_context_size = value.get("search_context_size", "medium")
        count = ZAI_WEB_SEARCH_COUNT_MAP.get(search_context_size, 10)

        # Map content_size based on search_context_size
        # low/medium → medium (shorter summaries), high → high (longer summaries)
        content_size: Literal["medium", "high"] = (
            "high" if search_context_size == "high" else "medium"
        )

        web_search_config: ZAIWebSearchConfig = {
            "search_engine": "search_pro_jina",
            "enable": True,
            "count": count,
            "search_recency_filter": "noLimit",
            "content_size": content_size,
            "result_sequence": "after",
            "search_result": False,
            "require_search": False,
        }

        return ZAIWebSearchTool(
            type="web_search",
            web_search=web_search_config,
        )

    def _add_web_search_to_tools(
        self,
        tools: Optional[List[Dict[str, Any]]],
        web_search_tool: ZAIWebSearchTool
    ) -> List[Dict[str, Any]]:
        """
        Add web search tool to existing tools list.

        Ensures no duplicate web_search tools are added.
        """
        if tools is None:
            tools = []

        # Check if web_search tool already exists
        has_web_search = any(
            tool.get("type") == "web_search" for tool in tools
        )

        if not has_web_search:
            tools.append(dict(web_search_tool))

        return tools

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI-style parameters to Z.AI format.

        Handles web_search_options by adding a web_search tool to the tools array.
        """
        # First, call parent to handle standard params
        optional_params = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

        # Handle web_search_options
        web_search_options = non_default_params.get("web_search_options")
        if web_search_options and isinstance(web_search_options, dict):
            # Map to Z.AI web search tool format
            web_search_tool = self._map_web_search_options(web_search_options)

            # Add to tools array
            existing_tools = optional_params.get("tools", [])
            optional_params["tools"] = self._add_web_search_to_tools(
                existing_tools, web_search_tool
            )

        return optional_params
