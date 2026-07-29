"""
Groq-specific cost helpers.

Groq bills built-in browser search per executed search ($5 per 1k searches,
https://groq.com/pricing), so web search cost scales with the number of
`browser.search` executions counted into `usage.prompt_tokens_details`.
"""

from typing import TYPE_CHECKING

from litellm.types.utils import PromptTokensDetailsWrapper, Usage

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo


def cost_per_web_search_request(usage: Usage, model_info: "ModelInfo") -> float:
    search_costs = model_info.get("search_context_cost_per_query") or {}
    cost_per_search = search_costs.get("search_context_size_medium", 0.0)
    if (
        usage is not None
        and usage.prompt_tokens_details is not None
        and isinstance(usage.prompt_tokens_details, PromptTokensDetailsWrapper)
        and getattr(usage.prompt_tokens_details, "web_search_requests", None) is not None
    ):
        return cost_per_search * usage.prompt_tokens_details.web_search_requests
    return 0.0
