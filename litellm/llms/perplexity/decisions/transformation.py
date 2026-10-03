from typing import Final

from litellm.llms.base_llm.decisions.transformation import JevCompatibleDecisionsEndpoint

PERPLEXITY_DECISIONS_ENDPOINT: Final[JevCompatibleDecisionsEndpoint] = JevCompatibleDecisionsEndpoint(
    default_api_base_value="https://api.perplexity.ai",
    path="/v1/decisions",
    api_key_env=("PERPLEXITYAI_API_KEY", "PERPLEXITY_API_KEY"),
    api_base_env="PERPLEXITY_API_BASE",
)
