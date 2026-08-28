"""
Groq-specific cost helpers.

Groq bills the built-in browser tool per executed action
(https://groq.com/pricing): `browser.search` at $5 per 1k and
`browser.open` at $1 per 1k. The Groq chat transformation counts both
action kinds off `executed_tools` into `usage.server_tool_use`.
"""

from typing import TYPE_CHECKING, Final, Mapping

from litellm.constants import GROQ_BROWSER_VISIT_WEBSITE_COST_PER_CALL
from litellm.types.utils import Usage

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo


def cost_per_web_search_request(usage: Usage, model_info: "ModelInfo") -> float:
    search_costs: Final = model_info.get("search_context_cost_per_query")
    if isinstance(search_costs, Mapping):
        cost_per_search: Final = float(search_costs.get("search_context_size_medium", 0.0) or 0.0)
    elif isinstance(search_costs, (int, float)):
        cost_per_search = float(search_costs)
    else:
        cost_per_search = 0.0

    server_tool_use: Final = getattr(usage, "server_tool_use", None)
    if server_tool_use is None:
        return 0.0
    searches: Final = server_tool_use.web_search_requests or 0
    opens: Final = server_tool_use.browser_open_requests or 0
    return searches * cost_per_search + opens * GROQ_BROWSER_VISIT_WEBSITE_COST_PER_CALL
