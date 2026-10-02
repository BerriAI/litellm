from typing import Final

from litellm.llms.base_llm.decisions.transformation import JevCompatibleDecisionsEndpoint

OPENROUTER_DECISIONS_ENDPOINT: Final[JevCompatibleDecisionsEndpoint] = JevCompatibleDecisionsEndpoint(
    default_api_base_value="https://openrouter.ai/api",
    path="/alpha/decisions",
    api_key_env=("OPENROUTER_API_KEY",),
    api_base_env="OPENROUTER_API_BASE",
)
