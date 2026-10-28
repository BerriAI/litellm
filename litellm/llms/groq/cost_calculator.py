"""
Groq-specific cost helpers.

Groq bills the built-in browser tool per executed action
(https://groq.com/pricing): `browser.search` at $5 per 1k and
`browser.open` at $1 per 1k. The Groq chat transformation counts both
action kinds off `executed_tools` into `usage.server_tool_use`.
"""

from typing import TYPE_CHECKING, Final

from litellm.types.utils import Usage

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo


def cost_per_web_search_request(usage: Usage, model_info: "ModelInfo") -> float:
    from litellm._cost import Usage as TokenUsage
    from litellm.litellm_core_utils.llm_cost_calc.utils import search_cost

    try:
        server_tool_use: Final = usage.server_tool_use
    except AttributeError:
        return 0.0
    if server_tool_use is None:
        return 0.0
    return search_cost(
        model_info,
        TokenUsage(
            prompt_tokens=0,
            text=0,
            output_text=0,
            web_search_requests=server_tool_use.web_search_requests or 0,
            browser_opens=server_tool_use.browser_open_requests or 0,
        ),
    )
