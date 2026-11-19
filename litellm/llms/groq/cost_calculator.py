"""
Groq-specific cost helpers.

Groq bills the built-in browser tool per executed action
(https://groq.com/pricing): `browser.search` at the row's search price and
`browser.open` per page visit. The Groq chat transformation counts both
action kinds off `executed_tools` into `usage.server_tool_use`.
"""

from typing import TYPE_CHECKING

from litellm.types.utils import Usage

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo


def cost_per_web_search_request(usage: Usage, model_info: "ModelInfo") -> float:
    """Groq's browser tool fee, priced in Rust."""
    from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import price_web_search_dispatch

    return price_web_search_dispatch("groq", usage, model_info) or 0.0
