"""
Mistral-specific web search cost calculation.

Mistral bills the built-in web search connectors per call, independent of the
model and the response size: web_search at $30 per 1,000 calls ($0.03/call) and
web_search_premium at $50 per 1,000 calls ($0.05/call). The Conversations
transformation surfaces the total call count as
usage.prompt_tokens_details.web_search_requests (the field the shared web search
detection reads) and the premium share as usage.web_search_premium_requests.

https://mistral.ai/pricing/
"""

from typing import TYPE_CHECKING

from litellm.types.utils import PromptTokensDetailsWrapper

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo, Usage

MISTRAL_WEB_SEARCH_COST_PER_CALL: float = 30.0 / 1000.0
MISTRAL_WEB_SEARCH_PREMIUM_COST_PER_CALL: float = 50.0 / 1000.0


def cost_per_web_search_request(usage: "Usage", model_info: "ModelInfo") -> float:
    """Cost of the web search calls made during a Mistral Conversations request."""
    details = getattr(usage, "prompt_tokens_details", None)
    total_requests = (
        int(details.web_search_requests)
        if isinstance(details, PromptTokensDetailsWrapper) and details.web_search_requests is not None
        else 0
    )
    premium_requests = int(getattr(usage, "web_search_premium_requests", 0) or 0)
    standard_requests = max(total_requests - premium_requests, 0)
    return (
        standard_requests * MISTRAL_WEB_SEARCH_COST_PER_CALL
        + premium_requests * MISTRAL_WEB_SEARCH_PREMIUM_COST_PER_CALL
    )
