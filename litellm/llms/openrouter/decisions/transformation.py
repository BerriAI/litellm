from typing import Final

from litellm.llms.base_llm.decisions.transformation import DecisionsEndpoint

OPENROUTER_DECISIONS_ENDPOINT: Final[DecisionsEndpoint] = DecisionsEndpoint(
    default_api_base="https://openrouter.ai/api",
    path="/alpha/decisions",
    api_key_env=("OPENROUTER_API_KEY",),
    api_base_env="OPENROUTER_API_BASE",
)
