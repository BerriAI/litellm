"""
Advisor interception module.

Provides completion/chat-completions advisor orchestration for providers that
do not natively support Anthropic's ``advisor_20260301`` tool type.
"""

from litellm.integrations.advisor_interception.handler import (
    AdvisorInterceptionLogger,
)
from litellm.integrations.advisor_interception.tools import (
    get_litellm_advisor_tool,
    get_litellm_advisor_tool_openai,
    is_advisor_tool,
)

__all__ = [
    "AdvisorInterceptionLogger",
    "get_litellm_advisor_tool",
    "get_litellm_advisor_tool_openai",
    "is_advisor_tool",
]
