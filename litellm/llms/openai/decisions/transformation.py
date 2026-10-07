from typing import Final

from litellm.llms.base_llm.decisions.transformation import JevCompatibleDecisionsEndpoint

OPENAI_DECISIONS_ENDPOINT: Final = JevCompatibleDecisionsEndpoint(
    default_api_base_value="https://api.openai.com/v1",
    path="/v1/decisions",
    api_key_env=("OPENAI_API_KEY",),
    api_base_env="OPENAI_API_BASE",
)
